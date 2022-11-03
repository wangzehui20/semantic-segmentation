import cv2
import os
import os.path as osp
import numpy as np
from tqdm import tqdm


def fast_hist(pred, gt, nclass=2):
    mask = (gt >= 0) & (gt < nclass)
    hist = np.bincount(
        nclass * gt[mask].astype(int) +
        pred[mask], minlength=nclass ** 2).reshape(nclass, nclass)
    return hist

def iou(hist):
    eps = 1e-8
    intersection = hist[-1,-1]
    union = np.sum(hist[-1,:]) + np.sum(hist[:,-1]) - intersection + eps
    iou = intersection / union
    return iou

def compare_iou(mask_path, pred_path, mask_update_path):
    label_map = {0:0, 255:1}
    mask = cv2.imread(mask_path, 0)
    pred = cv2.imread(pred_path, 0)
    mask_update = cv2.imread(mask_update_path, 0)
    for k, v in label_map.items():
        mask[mask == k] = v
        pred[pred == k] = v
        mask_update[mask_update == k] = v
    # hist
    pred_hist = fast_hist(pred, mask)
    mask_update_hist = fast_hist(mask_update, mask)
    # iou
    pred_iou = iou(pred_hist)
    mask_update_iou = iou(mask_update_hist)
    if pred_iou - mask_update_iou > 0.3:
        print(mask_update_path)

def compare_iou_dir(mask_dir, pred_dir, mask_update_dir):
    pred_names = sorted(os.listdir(pred_dir))
    for name in tqdm(pred_names, total=len(pred_names)):
        mask_path = osp.join(mask_dir, name)
        pred_path = osp.join(pred_dir, name)
        mask_update_path = osp.join(mask_update_dir, name)
        compare_iou(mask_path, pred_path, mask_update_path)


if __name__ == '__main__':
    mask_dir = r'/data/data/change_detection/merge/256_128/2016/mask'
    pred_dir = r'/data/data/change_detection/models/cyclegan/unet/effb1_dicebce/pred'
    mask_update_dir = r'/data/data/change_detection/models/cyclegan/unet/effb1_dicebce/mask_merge'
    compare_iou_dir(mask_dir, pred_dir, mask_update_dir)