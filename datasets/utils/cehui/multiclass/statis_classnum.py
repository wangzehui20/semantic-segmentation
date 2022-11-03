import os
import os.path as osp
import cv2
import numpy as np
import csv
from tqdm import tqdm


def create_classdict(classes):
    classdict = dict()
    for cls in classes:
        classdict[cls] = 0
    return classdict


def statis_classnum(labeldir, classes, csvpath):
    classdict = create_classdict(classes)
    label_list = os.listdir(labeldir)
    for label in tqdm(label_list, total=len(label_list)):
        labelpath = osp.join(labeldir, label)
        label = cv2.imread(labelpath, 0)
        for cls in classes:
            if np.any(label==cls):
                classdict[cls] += 1
    writercsv(csvpath, classdict)


def writercsv(logpath, data):
        firstline = []
        logline = []
        for k, v in data.items():
            if isinstance(v, float):
                v = "%.6f" % (v)
                firstline.append(k)
                logline.append(v)
            elif isinstance(v, str):
                firstline.append(k)
                logline.append(v)
            elif isinstance(v, int):
                firstline.append(k)
                logline.append(v)


        if os.path.isfile(logpath):
            with open(logpath, 'a', newline='')as f:
                csv_write = csv.writer(f, dialect='excel')
                csv_write.writerow(logline)
        else:
            with open(logpath, 'w', newline='')as f:
                csv_write = csv.writer(f, dialect='excel')
                csv_write.writerow(firstline)
                csv_write.writerow(logline)


if __name__ == '__main__':
    labeldir = r'/data/data/multiclass/train/labels'
    csvpath = r'/data/data/multiclass/class_num.csv'
    classes = [0,1,2,3,4,5,6,7,8]
    statis_classnum(labeldir, classes, csvpath)