import json
import os
import numpy as np
import pickle
import random
import shutil
from collections import Counter

REMOVE_THRED = 0.2   # if background ratio > thred, remove this image


def is_dir(dir):
    if not os.path.exists(dir): os.makedirs(dir)


def seg_double2int(segmentation):
    return [[int(s + 0.5) for seg in segmentation for s in seg]]


def bbox_double2int(bbox):
    return [int(bb + 0.5) for bb in bbox]


def open_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path, result):
    with open(path, 'w') as f:
        json.dump(result, f)


def save_pickle(path, result):
    with open(path, 'wb') as f:
        pickle.dump(result, f)


def imfilter(file):
    return True if file[-4:] in ['.tif', '.img'] else False


def get_imlist(imdir):
    imlist_all = os.listdir(imdir)
    imlist = list(filter(imfilter, imlist_all))
    return imlist


def is_lowimg(img):
    total = img.shape[0] * img.shape[1] * img.shape[2]
    counter = Counter(img.ravel())
    if counter[0] / total > REMOVE_THRED:
        return True
    return False


def get_mean_std(img):
    means, stdevs = [], []
    img = img[:, :, :, np.newaxis]
    img_band = img.shape[2]

    for i in range(img_band):
        pixels = img[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()
    stdevs.reverse()
    return [means, stdevs]


def split_train_val(seed, ratio, num):
    num_list = [n for n in range(num)]
    random.seed(seed)
    random.shuffle(num_list)
    split_idx = int(len(num_list) * ratio)
    train_list = num_list[:split_idx]
    val_list = num_list[split_idx:]
    return train_list, val_list


def move_val_img(trainimg_dir, valimg_dir, val_list):
    img_list = os.listdir(trainimg_dir)
    for i, img in enumerate(img_list):
        if i in val_list:
            oripath = os.path.join(trainimg_dir, img)
            dstpath = os.path.join(valimg_dir, img)
            shutil.move(oripath, dstpath)
