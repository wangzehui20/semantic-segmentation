import json
import numpy as np
import os
import os.path as osp
import cv2
import math
from collections import Counter
from tqdm import tqdm


def normalization(dt):
    _range = np.max(dt) - np.min(dt)
    return (dt - np.min(dt)) / _range


def _norm(dt):
    _sum = np.sum(dt)
    return dt / _sum


def invlog_func(pixel_weights):
    invlog_pixel_weights = []
    for pw in pixel_weights:
        if pw <= 1:
            invlog_pixel_weights.append(0)
        else:
            invlog_pixel_weights.append(1 / math.log(pw))
    return _norm(invlog_pixel_weights)


def cal_class_weights(maskdir, classes, ignore_class):
    data = dict()
    names = os.listdir(maskdir)
    for name in tqdm(names, total=len(names)):
        pixel_weights = []
        img = cv2.imread(osp.join(maskdir, name), 0)
        counter = Counter(img.ravel())
        for i in range(classes):
            if i in ignore_class:
                pixel_weights[0] += counter[i]
            else:
                if i in counter:
                    pixel_weights.append(counter[i])
                else:
                    pixel_weights.append(0)
        data[name] = invlog_func(pixel_weights).tolist()
    return data


def dump_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    classes = 16
    ignore_class = [8, 9, 15]
    maskdir = r'/data/data/semi_compete/clip_integrate/512_128/labeled_train/mask'
    jsonpath = r'/data/data/semi_compete/clip_integrate/512_128/labeled_train/class_weights.json'
    data = cal_class_weights(maskdir, classes, ignore_class)
    dump_json(jsonpath, data)
