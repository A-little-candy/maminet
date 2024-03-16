import random

import cv2
import numpy as np
import torch
from scipy.ndimage.morphology import distance_transform_edt
from scipy.signal import convolve2d
from config import Config
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str, cfg: Config, kind: str):
        super(Dataset, self).__init__()
        self.path: str = path
        self.cfg: Config = cfg
        self.kind: str = kind
        self.image_size: (int, int) = (self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT)
        self.grayscale: bool = self.cfg.INPUT_CHANNELS == 1

        self.num_negatives_per_one_positive: int = 1
        self.frequency_sampling: bool = self.cfg.FREQUENCY_SAMPLING and self.kind == 'TRAIN'


    def init_extra(self):
        self.counter = 0
        self.neg_imgs_permutation = np.random.permutation(self.num_neg)
        self.pos_imgs_permutation = np.random.permutation(self.num_neg)

        self.neg_retrieval_freq = np.zeros(shape=self.num_neg)
        self.pos_retrieval_freq = np.zeros(shape=self.num_pos)

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor, bool, str):

        if self.counter >= self.len:
            self.counter = 0
            if self.frequency_sampling:
                sample_probability = 1 - (self.neg_retrieval_freq / np.max(self.neg_retrieval_freq))
                sample_probability = sample_probability - np.median(sample_probability) + 1
                sample_probability = sample_probability ** (np.log(len(sample_probability)) * 4)
                sample_probability = sample_probability / np.sum(sample_probability)

                # use replace=False for to get only unique values
                self.neg_imgs_permutation = np.random.choice(range(self.num_neg),
                                                             size=self.num_negatives_per_one_positive * self.num_pos,
                                                             p=sample_probability,
                                                             replace=False)
            else:
                self.neg_imgs_permutation = np.random.permutation(self.num_neg)

            self.pos_imgs_permutation = np.random.permutation(self.num_pos)

        if self.kind == 'TRAIN':
            if index >= self.num_pos:
                ix = index % self.num_pos
                ix = self.neg_imgs_permutation[ix]
                item = self.neg_samples[ix]
                self.neg_retrieval_freq[ix] = self.neg_retrieval_freq[ix] + 1

            else:
                ix = index
                item = self.pos_samples[ix]
        else:
            if index < self.num_neg:
                ix = index
                item = self.neg_samples[ix]
            else:
                ix = index - self.num_neg
                item = self.pos_samples[ix]

        image, seg_mask, seg_loss_mask, is_segmented, image_path, seg_mask_path, sample_name = item

        if self.cfg.ON_DEMAND_READ:  # STEEL only so far
            if image_path == -1 or seg_mask_path == -1:
                raise Exception('For ON_DEMAND_READ image and seg_mask paths must be set in read_contents')
            img = self.read_img_resize(image_path, self.grayscale, self.image_size)
            if seg_mask_path is None:  # good sample
                seg_mask = np.zeros_like(img)
            elif isinstance(seg_mask_path, list):
                seg_mask = self.rle_to_mask(seg_mask_path, self.image_size)
            else:
                seg_mask, _ = self.self.read_label_resize(seg_mask_path, self.image_size)

            if np.max(seg_mask) == np.min(seg_mask):  # good sample
                seg_loss_mask = np.ones_like(seg_mask)
            else:
                seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)

            # if self.kind == 'TRAIN':
            #     if random.random() < 0.5:
            #         img, seg_mask, seg_loss_mask = self.jigsaw_generator(img,seg_mask,seg_loss_mask,8)


            image = self.to_tensor(img)
            seg_mask = self.to_tensor(self.downsize(seg_mask))
            seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))

        self.counter = self.counter + 1

        # if self.kind == 'TRAIN':
        #     image, seg_mask, seg_loss_mask = self.DataAugment(image, seg_mask, seg_loss_mask)

        return image, seg_mask, seg_loss_mask, is_segmented, sample_name



    def __len__(self):
        return self.len

    # def jigsaw_generator(self,images,seg_mask, seg_loss_mask, n):
    #     l = []
    #     for a in range(n):
    #         for b in range(n):
    #             l.append([a, b])
    #     block_size_x = self.image_size[0] // n
    #     block_size_y = self.image_size[1] // n
    #     rounds = n ** 2
    #     random.shuffle(l)
    #     jigsaws_img = images.clone()
    #     jigsaws_seg_mask = seg_mask.clone()
    #     jigsaws_seg_loss_mask = seg_loss_mask.clone()
    #
    #     for i in range(rounds):
    #         x, y = l[i]
    #         temp_img = jigsaws_img[..., 0:block_size_x, 0:block_size_y].clone()
    #         temp_seg_mask = jigsaws_seg_mask[..., 0:block_size_x, 0:block_size_y].clone()
    #         temp_seg_loss_mask = jigsaws_seg_mask[..., 0:block_size_x, 0:block_size_y].clone()
    #
    #         jigsaws_img[..., 0:block_size_x, 0:block_size_y] = jigsaws_img[..., x * block_size_x:(x + 1) * block_size_x,
    #                                                    y * block_size_y:(y + 1) * block_size_y].clone()
    #         jigsaws_seg_mask[..., 0:block_size_x, 0:block_size_y] = jigsaws_seg_mask[..., x * block_size_x:(x + 1) * block_size_x,
    #                                                            y * block_size_y:(y + 1) * block_size_y].clone()
    #         jigsaws_seg_loss_mask[..., 0:block_size_x, 0:block_size_y] = jigsaws_seg_loss_mask[..., x * block_size_x:(x + 1) * block_size_x,
    #                                                            y * block_size_y:(y + 1) * block_size_y].clone()
    #
    #         jigsaws_img[..., x * block_size_x:(x + 1) * block_size_x, y * block_size_y:(y + 1) * block_size_y] = temp_img
    #         jigsaws_seg_mask[..., x * block_size_x:(x + 1) * block_size_x,
    #         y * block_size_y:(y + 1) * block_size_y] = temp_seg_mask
    #         jigsaws_seg_loss_mask[..., x * block_size_x:(x + 1) * block_size_x,
    #         y * block_size_y:(y + 1) * block_size_y] = temp_seg_loss_mask
    #
    #     return jigsaws_img,jigsaws_seg_mask,jigsaws_seg_loss_mask

    def DataAugment(self, image, seg_mask, seg_loss_mask):
        if random.random() < 0.5:
            HF = transforms.RandomHorizontalFlip(p = 1)
            image = HF(image)
            seg_mask = HF(seg_mask)
            seg_loss_mask = HF(seg_loss_mask)
        #
        # if random.random() < 0.25:
        #     VF = transforms.RandomVerticalFlip(p = 1)
        #     image = VF(image)
        #     seg_mask = VF(seg_mask)
        #     seg_loss_mask = VF(seg_loss_mask)


        # if random.random() < 0.5:
        #     angle = random.randint(0, 90)
        #     matRotate = cv2.getRotationMatrix2D((resize_dim[0] * 0.5, resize_dim[0] * 0.5), angle, 0.7)
        #     image = cv2.warpAffine(image, matRotate, resize_dim)
        #     seg_mask = cv2.warpAffine(seg_mask, matRotate, resize_dim)
        #     seg_loss_mask = cv2.warpAffine(seg_loss_mask, matRotate, resize_dim)


        return image, seg_mask, seg_loss_mask



    def read_contents(self):
        pass

    def read_img_resize(self, path, grayscale, resize_dim) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if resize_dim is not None:
            img = cv2.resize(img, dsize=resize_dim)
        return np.array(img, dtype=np.float32) / 255.0

    def read_label_resize(self, path, resize_dim, dilate=None) -> (np.ndarray, bool):
        lbl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if dilate is not None and dilate > 1:
            lbl = cv2.dilate(lbl, np.ones((dilate, dilate)))
        if resize_dim is not None:
            lbl = cv2.resize(lbl, dsize=resize_dim)
        return np.array((lbl / 255.0), dtype=np.float32), np.max(lbl) > 0

    def to_tensor(self, x) -> torch.Tensor:
        # if np.max(x) > 1.0:
        if x.dtype != np.float32:
            x = (x / 255.0).astype(np.float32)

        if len(x.shape) == 3:
            x = np.transpose(x, axes=(2, 0, 1))
        else:
            x = np.expand_dims(x, axis=0)

        x = torch.from_numpy(x)
        return x

    def distance_transform(self, mask: np.ndarray, max_val: float, p: float) -> np.ndarray:
        dst_trf = distance_transform_edt(mask)

        if dst_trf.max() > 0:
            dst_trf = (dst_trf / dst_trf.max())
            dst_trf = (dst_trf ** p) * max_val

        dst_trf[mask == 0] = 1.0
        return np.array(dst_trf, dtype=np.float32)

    def downsize(self, image: np.ndarray, downsize_factor: int = 8) -> np.ndarray:
        img_t = torch.from_numpy(np.expand_dims(image, 0 if len(image.shape) == 3 else (0, 1)).astype(np.float32))
        image_np = img_t.detach().numpy()
        # img_t = torch.from_numpy(np.expand_dims(image, 0).astype(np.float32))
        # img_t = torch.nn.ReflectionPad2d(padding=(downsize_factor))(img_t)
        # image_np = torch.nn.AvgPool2d(kernel_size=2 * downsize_factor + 1, stride=downsize_factor)(img_t).detach().numpy()
        return image_np[0] if len(image.shape) == 3 else image_np[0, 0]

    def rle_to_mask(self, rle, image_size):
        if len(rle) % 2 != 0:
            raise Exception('Suspicious')

        w, h = image_size
        mask_label = np.zeros(w * h, dtype=np.float32)

        positions = rle[0::2]
        length = rle[1::2]
        for pos, le in zip(positions, length):
            mask_label[pos - 1:pos + le - 1] = 1
        mask = np.reshape(mask_label, (h, w), order='F')
        return mask
