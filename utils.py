import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import pandas as pd
import os
import errno
import pickle
import cv2
import ipdb
import random
import torch

def jigsaw_generator( images, seg_mask, seg_loss_mask, n):

    image_size = images.shape[2:]
    # ipdb.set_trace()
    if n != 1:
        l = []
        for a in range(n):
            for b in range(n):
                l.append([a, b])
        block_size_x = image_size[0] // n
        block_size_y = image_size[1] // n
        rounds = n ** 2
        random.shuffle(l)
        jigsaws_img = images.clone()
        jigsaws_seg_mask = seg_mask.clone()
        jigsaws_seg_loss_mask = seg_loss_mask.clone()

        for i in range(rounds):
            x, y = l[i]
            temp_img = jigsaws_img[..., 0:block_size_x, 0:block_size_y].clone()
            temp_seg_mask = jigsaws_seg_mask[..., 0:block_size_x, 0:block_size_y].clone()
            temp_seg_loss_mask = jigsaws_seg_mask[..., 0:block_size_x, 0:block_size_y].clone()

            jigsaws_img[..., 0:block_size_x, 0:block_size_y] = jigsaws_img[..., x * block_size_x:(x + 1) * block_size_x,
                                                               y * block_size_y:(y + 1) * block_size_y].clone()
            jigsaws_seg_mask[..., 0:block_size_x, 0:block_size_y] = jigsaws_seg_mask[...,
                                                                    x * block_size_x:(x + 1) * block_size_x,
                                                                    y * block_size_y:(y + 1) * block_size_y].clone()
            jigsaws_seg_loss_mask[..., 0:block_size_x, 0:block_size_y] = jigsaws_seg_loss_mask[...,
                                                                         x * block_size_x:(x + 1) * block_size_x,
                                                                         y * block_size_y:(y + 1) * block_size_y].clone()

            jigsaws_img[..., x * block_size_x:(x + 1) * block_size_x, y * block_size_y:(y + 1) * block_size_y] = temp_img
            jigsaws_seg_mask[..., x * block_size_x:(x + 1) * block_size_x,
            y * block_size_y:(y + 1) * block_size_y] = temp_seg_mask
            jigsaws_seg_loss_mask[..., x * block_size_x:(x + 1) * block_size_x,
            y * block_size_y:(y + 1) * block_size_y] = temp_seg_loss_mask

        if n == 8:
            jigsaws_seg_mask = downsize(jigsaws_seg_mask, 2)
            jigsaws_seg_loss_mask = downsize(jigsaws_seg_loss_mask, 2)
        elif n == 4:
            jigsaws_seg_mask = downsize(jigsaws_seg_mask, 4)
            jigsaws_seg_loss_mask = downsize(jigsaws_seg_loss_mask, 4)
        elif n == 2:
            jigsaws_seg_mask = downsize(jigsaws_seg_mask, 8)
            jigsaws_seg_loss_mask = downsize(jigsaws_seg_loss_mask, 8)
    else:
        jigsaws_img = images
        jigsaws_seg_mask = downsize(seg_mask, 8)
        jigsaws_seg_loss_mask = downsize(seg_loss_mask, 8)

    return jigsaws_img, jigsaws_seg_mask, jigsaws_seg_loss_mask


def downsize(image, downsize_factor):
    # img_t = torch.from_numpy(np.expand_dims(image, 0 if len(image.shape) == 3 else (0, 1)).astype(np.float32))
    # img_t = torch.from_numpy(np.expand_dims(image, 0).astype(np.float32))
    img_t = torch.nn.ReflectionPad2d(padding=(downsize_factor))(image)
    image_np = torch.nn.AvgPool2d(kernel_size=2 * downsize_factor + 1, stride=downsize_factor)(img_t)
    return image_np


def create_folder(folder, exist_ok=True):
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST or not exist_ok:
            raise


def calc_confusion_mat(D, Y):
    FP = (D != Y) & (Y.astype(np.bool) == False)
    FN = (D != Y) & (Y.astype(np.bool) == True)
    TN = (D == Y) & (Y.astype(np.bool) == False)
    TP = (D == Y) & (Y.astype(np.bool) == True)

    return FP, FN, TN, TP


def plot_sample(image_name, image, segmentation, label, save_dir, decision=None, blur=True, plot_seg=False):
    plt.figure()
    plt.clf()
    plt.subplot(1, 4, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Input image')
    if image.shape[0] < image.shape[1]:
        image = np.transpose(image, axes=[1, 0, 2])
        segmentation = np.transpose(segmentation)
        label = np.transpose(label)
    if image.shape[2] == 1:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    plt.subplot(1, 4, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Groundtruth')
    plt.imshow(label, cmap="gray")

    plt.subplot(1, 4, 3)
    plt.xticks([])
    plt.yticks([])
    if decision is None:
        plt.title('Output')
    else:
        plt.title(f"Output: {decision:.5f}")
    # display max
    vmax_value = max(1, np.max(segmentation))
    plt.imshow(segmentation, cmap="jet", vmax=vmax_value)

    plt.subplot(1, 4, 4)
    plt.xticks([])
    plt.yticks([])
    plt.title('Output scaled')
    if blur:
        normed = segmentation / segmentation.max()
        blured = cv2.blur(normed, (32, 32))
        plt.imshow((blured / blured.max() * 255).astype(np.uint8), cmap="jet")
    else:
        plt.imshow((segmentation / segmentation.max() * 255).astype(np.uint8), cmap="jet")

    out_prefix = '{:.3f}_'.format(decision) if decision is not None else ''

    plt.savefig(f"{save_dir}/{out_prefix}result_{image_name}.jpg", bbox_inches='tight', dpi=300)
    plt.close()

    if plot_seg:
        jet_seg = cv2.applyColorMap((segmentation * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f"{save_dir}/{out_prefix}_segmentation_{image_name}.png", jet_seg)


def evaluate_metrics(samples, results_path, run_name):
    samples = np.array(samples)

    img_names = samples[:, 4]
    predictions = samples[:, 0]
    labels = samples[:, 3].astype(np.float32)

    metrics = get_metrics(labels, predictions)

    df = pd.DataFrame(
        data={'prediction': predictions,
              'decision': metrics['decisions'],
              'ground_truth': labels,
              'img_name': img_names})
    df.to_csv(os.path.join(results_path, 'results.csv'), index=False)

    print(
        f'{run_name} EVAL AUC={metrics["AUC"]:f}, and AP={metrics["AP"]:f}, w/ best thr={metrics["best_thr"]:f} at f-m={metrics["best_f_measure"]:.3f} and FP={sum(metrics["FP"]):d}, FN={sum(metrics["FN"]):d}')

    with open(os.path.join(results_path, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
        f.close()

    plt.figure(1)
    plt.clf()
    plt.plot(metrics['recall'], metrics['precision'])
    plt.title('Average Precision=%.4f' % metrics['AP'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f"{results_path}/precision-recall.pdf", bbox_inches='tight')

    plt.figure(1)
    plt.clf()
    plt.plot(metrics['FPR'], metrics['TPR'])
    plt.title('AUC=%.4f' % metrics['AUC'])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig(f"{results_path}/ROC.pdf", bbox_inches='tight')


def get_metrics(labels, predictions):
    metrics = {}
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['thresholds'] = thresholds
    f_measures = 2 * np.multiply(recall, precision) / (recall + precision + 1e-8)
    metrics['f_measures'] = f_measures
    ix_best = np.argmax(f_measures)
    metrics['ix_best'] = ix_best
    best_f_measure = f_measures[ix_best]
    metrics['best_f_measure'] = best_f_measure
    best_thr = thresholds[ix_best]
    metrics['best_thr'] = best_thr
    FPR, TPR, _ = roc_curve(labels, predictions)
    metrics['FPR'] = FPR
    metrics['TPR'] = TPR
    AUC = auc(FPR, TPR)
    metrics['AUC'] = AUC
    AP = auc(recall, precision)
    metrics['AP'] = AP
    decisions = predictions >= best_thr
    metrics['decisions'] = decisions
    FP, FN, TN, TP = calc_confusion_mat(decisions, labels)
    metrics['FP'] = FP
    metrics['FN'] = FN
    metrics['TN'] = TN
    metrics['TP'] = TP
    metrics['accuracy'] = (sum(TP) + sum(TN)) / (sum(TP) + sum(TN) + sum(FP) + sum(FN))
    return metrics
