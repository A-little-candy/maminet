import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from functools import partial
# from bn_lib.nn.modules import SynchronizedBatchNorm2d
BATCHNORM_TRACK_RUNNING_STATS = False
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997
from torch.nn.modules.batchnorm import _BatchNorm

class External_attention(nn.Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''

    def __init__(self, c):
        super(External_attention, self).__init__()

        self.conv1 = nn.Conv2d(c, c, 1)

        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h * w
        x = x.view(b, c, h * w)  # b * c * n

        # 计算注意力分数
        attn = self.linear_0(x)  # b, k, n
        attn = F.softmax(attn, dim=-1)  # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n
        x = self.linear_1(attn)  # b, c, n

        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x)
        return x

class BNorm_init(nn.BatchNorm2d):
    def reset_parameters(self):
        init.uniform_(self.weight, 0, 1)
        init.zeros_(self.bias)

# 定义2d卷积块
class Conv2d_init(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super(Conv2d_init, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def reset_parameters(self):
        # 一种常见的初始化权重矩阵方法，pytorch提供的初始化函数之一
        init.xavier_normal_(self.weight)
        if self.bias is not None:
            # 获取权重的输入通道数，并对偏置项进行均匀初始化，保证参数在一个合理范围内
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

# 定义一个大的卷积块，包括卷积层，特征归一化以及ReLU
def _conv_block(in_chanels, out_chanels, kernel_size, padding):
    return nn.Sequential(Conv2d_init(in_channels=in_chanels, out_channels=out_chanels,
                                     kernel_size=kernel_size, padding=padding, bias=False),
                         FeatureNorm(num_features=out_chanels, eps=0.001),
                         nn.ReLU())

# 定义了一个特征归一化层
class FeatureNorm(nn.Module):
    # feature_index，self.scale和self.bias到底什么用？
    def __init__(self, num_features, feature_index=1, rank=4, reduce_dims=(2, 3), eps=0.001, include_bias=True):
        # num_features：特征的数量，feature_index：在哪个维度上进行特征归一化，默认为第一维度，rank：数据的维度
        # reduce_dims：用于计算均值和标准差的维度，eps：一个小常数，用于防止除以0，include_bias:是否包含偏置项
        super(FeatureNorm, self).__init__()
        # 创建了一个长度为rank的列表，所有元素都为1
        self.shape = [1] * rank
        # 将特定维度的元素设置为num_features，表示该维度特征的数量
        self.shape[feature_index] = num_features
        self.reduce_dims = reduce_dims

        # 创建一个可学习的参数self.scale，是一个self.shape的张量，初始值全为1
        self.scale = nn.Parameter(torch.ones(self.shape, requires_grad=True, dtype=torch.float))
        # 创建一个可学习的参数self.bias，如果包含偏置项，这个参数requires_grad，否则不requires_grad
        self.bias = nn.Parameter(torch.zeros(self.shape, requires_grad=True, dtype=torch.float)) if include_bias else nn.Parameter(
            torch.zeros(self.shape, requires_grad=False, dtype=torch.float))

        self.eps = eps

    def forward(self, features):
        # 计算self.reduce_dims维度的标准差和均值
        f_std = torch.std(features, dim=self.reduce_dims, keepdim=True)
        f_mean = torch.mean(features, dim=self.reduce_dims, keepdim=True)
        # 计算标准化的特征并将特征进行缩放和平移
        return self.scale * ((features - f_mean) / (f_std + self.eps).sqrt()) + self.bias


class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        # self.conv4 = nn.Conv2d(64,64,3,padding=1)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.relu4 = nn.ReLU(inplace=True)
        #
        # self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(64,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(64,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        # self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        # self.bn_d2 = nn.BatchNorm2d(64)
        # self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(64,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        # hx4 = self.relu4(self.bn4(self.conv4(hx)))
        # hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx1)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(hx)))
        hx = self.upscore2(d4)

        # ipdb.set_trace()
        d3 = self.relu_d3(self.bn_d3(self.conv_d3(hx)))
        hx = self.upscore2(d3)

        # d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        # hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(hx)))

        # ipdb.set_trace()
        residual = self.conv_d0(d1)

        return residual


class SegDecNet(nn.Module):
    def __init__(self, device, input_width, input_height, input_channels):
        super(SegDecNet, self).__init__()
        if input_width % 8 != 0 or input_height % 8 != 0:
            raise Exception(f"Input size must be divisible by 8! width={input_width}, height={input_height}")
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        # BFE
        self.volume = nn.Sequential(_conv_block(self.input_channels, 32, 5, 2),
                                    # _conv_block(32, 32, 5, 2), # Has been accidentally left out and remained the same since then
                                    nn.MaxPool2d(2),
                                    _conv_block(32, 64, 5, 2),
                                    _conv_block(64, 64, 5, 2),
                                    _conv_block(64, 64, 5, 2),
                                    nn.MaxPool2d(2),
                                    _conv_block(64, 64, 5, 2),
                                    _conv_block(64, 64, 5, 2),
                                    _conv_block(64, 64, 5, 2),
                                    _conv_block(64, 64, 5, 2),
                                    nn.MaxPool2d(2),
                                    #External_attention(64),
                                    _conv_block(64, 1024, 15, 7))

        self.seg_mask = nn.Sequential(
            Conv2d_init(in_channels=1024, out_channels=1, kernel_size=1, padding=0, bias=False),
            FeatureNorm(num_features=1, eps=0.001, include_bias=False))
        #self.seg_mask_refine = RefUnet(1,64)

        self.extractor = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                       _conv_block(in_chanels=1025, out_chanels=8, kernel_size=5, padding=2),
                                       nn.MaxPool2d(kernel_size=2),
                                       _conv_block(in_chanels=8, out_chanels=16, kernel_size=5, padding=2),
                                       nn.MaxPool2d(kernel_size=2),
                                       _conv_block(in_chanels=16, out_chanels=32, kernel_size=5, padding=2))

        self.global_max_pool_feat = nn.MaxPool2d(kernel_size=32)
        self.global_avg_pool_feat = nn.AvgPool2d(kernel_size=32)
        self.global_max_pool_seg = nn.MaxPool2d(kernel_size=(self.input_height / 8, self.input_width / 8))
        self.global_avg_pool_seg = nn.AvgPool2d(kernel_size=(self.input_height / 8, self.input_width / 8))

        self.fc = nn.Linear(in_features=66, out_features=1)


        self.volume_lr_multiplier_layer = GradientMultiplyLayer().apply
        self.glob_max_lr_multiplier_layer = GradientMultiplyLayer().apply
        self.glob_avg_lr_multiplier_layer = GradientMultiplyLayer().apply
        self.relu = nn.Sigmoid()
        
        self.device = device


    def set_gradient_multipliers(self, multiplier):
        self.volume_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        self.glob_max_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        self.glob_avg_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)

    def forward(self, input):
        #BFE
        volume = self.volume(input)
        
        #SD
        seg_mask = self.seg_mask(volume)
        #seg_mask = torch.zeros_like(seg_mask)
        cat = torch.cat([volume, seg_mask], dim=1)

        cat = self.volume_lr_multiplier_layer(cat, self.volume_lr_multiplier_mask)
        
        #MiAA
        features = self.extractor(cat)
        #features = volume
        global_max_feat = torch.max(torch.max(features, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        global_avg_feat = torch. mean(features, dim=(-1, -2), keepdim=True)
        global_max_seg = torch.max(torch.max(seg_mask, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        global_avg_seg = torch.mean(seg_mask, dim=(-1, -2), keepdim=True)

        global_max_feat = global_max_feat.reshape(global_max_feat.size(0), -1)
        global_avg_feat = global_avg_feat.reshape(global_avg_feat.size(0), -1)

        global_max_seg = global_max_seg.reshape(global_max_seg.size(0), -1)
        global_max_seg = self.glob_max_lr_multiplier_layer(global_max_seg, self.glob_max_lr_multiplier_mask)
        global_avg_seg = global_avg_seg.reshape(global_avg_seg.size(0), -1)
        global_avg_seg = self.glob_avg_lr_multiplier_layer(global_avg_seg, self.glob_avg_lr_multiplier_mask)

        #Classifier
        fc_in = torch.cat([global_max_feat, global_avg_feat, global_max_seg, global_avg_seg], dim=1)
        #fc_in = torch.cat([global_max_feat, global_avg_feat], dim=1)
        fc_in = fc_in.reshape(fc_in.size(0), -1)
        prediction = self.fc(fc_in)
        out_seg = self.relu(seg_mask)


        return prediction, seg_mask, out_seg



class GradientMultiplyLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask_bw):
        ctx.save_for_backward(mask_bw)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        mask_bw, = ctx.saved_tensors
        return grad_output.mul(mask_bw), None


