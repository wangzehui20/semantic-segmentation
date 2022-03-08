import csv
import os
import numpy as np
import os.path as osp
from typing import Counter
import pandas as pd
import cv2
from tqdm import tqdm


class Csv:
    def __init__(self) -> None:
        self.keys = []
        self.content = dict()
        
    def readcsv(self, filepath):
        with open(filepath, encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i == 0:
                    for key in line:
                        if key not in self.content.keys():
                            self.content[key] = []
                        self.keys.append(key)
                else:
                    for j, value in enumerate(line):
                        self.content[self.keys[j]].append(value)


    def write(self, logpath):
        num = len(self.content[self.keys[0]])
        for i in range(num):
            data = dict()
            for key in self.keys:
                data[key] = self.content[key][i]
            self.writecsv(logpath, data)

    def writecsv(self, logpath, data):
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


def record_classinfo(maskdir, names, classes, log_path):
    c = Csv()
    class_list = [[] for c in range(classes)]
    for name in tqdm(names, total=len(names)):
        path = osp.join(maskdir, name)
        img = cv2.imread(path, 0)
        data = cal_pixelratio(img, classes)
        for i, v in enumerate(data.values()):
            class_list[i].append(v)

        # data['name'] = name
        # c.writecsv(log_path, data)
    data = dict()
    for i in range(classes):
        data[i] = np.mean(class_list[i])
    c.writecsv(log_path, data)


def cal_pixelratio(img, classes):
    total_pixel = img.shape[0] * img.shape[1]
    data = dict()
    counter = Counter(img.ravel())
    for i in range(classes):
        data[i] = counter[i] / total_pixel
    return data


    

if __name__ == '__main__':
    # mean pixel ratio of all images

    # log_path = r'/data/data/semi_compete/clip_integrate/512_128/labeled_train/mean_pixelratio/train.csv'
    # maskdir = r'/data/data/semi_compete/clip_integrate/512_128/labeled_train/mask'
    # ids_csv = r'/data/data/semi_compete/clip_integrate/512_128/labeled_train/train.csv'
    # classes = 16   # include no data class
    # names = pd.read_csv(ids_csv)['name'].tolist() if ids_csv and osp.exists(ids_csv) else None
    # record_classinfo(maskdir, names, classes, log_path)


    # pixel ratio of one label

    log_path = r'/data/data/semi_compete/clip_integrate/512_128/labeled_train/pixelratio/train.csv'
    label_logpath = r'/data/data/semi_compete/clip_integrate/512_128/labeled_train/label_top/6_name.csv'
    label = 6
    c = Csv()
    c.readcsv(log_path)
    print(c.content.keys())
    for l, n in sorted(zip(c.content[str(label)], c.content['name']), key=lambda pair: pair[0], reverse=True):
        data = dict()
        data['name'] = n
        # data[label] = l
        c.writecsv(label_logpath, data)
        

