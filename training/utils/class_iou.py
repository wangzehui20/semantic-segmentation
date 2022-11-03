# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:50:48 2021

@author: DYP
"""

import cv2
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import multiprocessing

CLASSES = 9


def cal_iou(pre, gt, class_index):
    pre_class = pre == class_index
    gt_class = gt == class_index

    inter = np.sum(pre_class * gt_class)
    union = np.sum(pre_class) + np.sum(gt_class) - inter
    return inter, union


def getpiciou(pathes):
    imgpath, gtpath = pathes
    inters = np.zeros((CLASSES,))
    unions = np.zeros((CLASSES,))
    pre_img = convert_label(cv2.imread(imgpath, 0))
    gt_img = cv2.imread(gtpath, 0)
    for index in range(CLASSES):
        inter, union = cal_iou(pre_img, gt_img, index)
        inters[index] += inter
        unions[index] += union
    return inters, unions


def diriou(predir, gtdir):
    imlist = os.listdir(predir)

    ious = np.zeros((CLASSES,))
    inters = np.zeros((CLASSES,))
    unions = np.zeros((CLASSES,))

    inlist = []
    for imname in imlist:
        inlist.append([os.path.join(predir, imname), os.path.join(gtdir, imname)])

    with Pool() as p:
        all_rows = p.map(getpiciou, inlist)

        for inter, union in all_rows:
            inters += inter
            unions += union

    for index in range(CLASSES):
        ious[index] = inters[index] / unions[index]
        print(f"class{index}: {ious[index]}")
    # print("mean_iou", np.mean(ious))


def convert_label(label, inverse=False):
    label_mapping = {0: 0,
                    10: 1,
                    20: 2,
                    30: 3,
                    40: 4,
                    50: 5,
                    60: 6,
                    70: 7,
                    80: 8,}

    tmp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[tmp==k] = v
    else:
        for k, v in label_mapping.items():
            label[tmp==k] = v
        label[label>len(label_mapping)-1] = 0
    return label


if __name__ == '__main__':
    prepath = r"/data/data/multiclass/models/Unet_bifpn/effb3_ce/val_results_epoch36"
    gtpath = r"/data/data/multiclass/train/labels"
    diriou(prepath, gtpath)
