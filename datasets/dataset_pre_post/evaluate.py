# encoding: utf-8
import numpy as np
import os
import cv2
import os.path as osp
import csv
from tqdm import tqdm
from osgeo import gdal
from common import get_imlist


def caloneimg_score(labelmerge_dir, predmerge_dir, label_map, eva_csvpath):
    """
    calculate one big image macro score
    """
    classes = len(label_map)+1
    labelist = os.listdir(labelmerge_dir)
    e = Evaluate()
    c = Csv()
    for name in tqdm(labelist, total=len(labelist)):
        hist = np.zeros((classes, classes))
        labelpath = osp.join(labelmerge_dir, name)
        predpath = osp.join(predmerge_dir, name)
        label = cv2.imread(labelpath, 0)
        pred = cv2.imread(predpath[:-4]+'.img', 0)
        for k, v in label_map.items():
            label[label==k] = v
            pred[pred==k] = v
        label[label>len(label_map)] = 0
        pred[pred>len(label_map)] = 0
        hist = e.fast_hist(pred, label, classes)
        f1, f0_5, precision, recall = e.macroscore(hist)

        data = dict()
        data['name'] = name
        data['f1'] = f1
        c.writecsv(eva_csvpath, data)


def calimg_score(labelmerge_dir, predmerge_dir, label_map, eva_csvpath):
    """
    calculate all big images macro score
    """
    classes = len(label_map)+1
    labelist = get_imlist(labelmerge_dir)
    hist = np.zeros((classes, classes))
    e = Evaluate()
    c = Csv()
    for name in tqdm(labelist, total=len(labelist)):
        labelpath = osp.join(labelmerge_dir, name)
        predpath = osp.join(predmerge_dir, name)
        label = read_image2(labelpath)
        pred = read_image2(predpath)
        # label = cv2.imread(labelpath, 0)
        # pred = cv2.imread(predpath, 0)
        for k, v in label_map.items():
            label[label==k] = v
            pred[pred==k] = v
        label[label>len(label_map)] = 0
        pred[pred>len(label_map)] = 0
        hist += e.fast_hist(pred, label, classes)
    f1, f0_5, precision, recall = e.macroscore(hist)
    print(f1, f0_5, precision, recall)

    # data = dict()
    # data['name'] = name
    # data['f1'] = f1
    # c.writecsv(eva_csvpath, data)


def read_image2(filename):
    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    del dataset

    if len(im_data.shape) == 2:
        return im_data


class Csv:
    def __init__(self) -> None:
        self.keys = []
        self.content = dict()
        
    def readcsv(self, filepath, topk=0):
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
                if i == topk:
                    break

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


class Evaluate:
    """
    主要用来计算类别f1, precision, recall
    """
    def __init__(self) -> None:
        pass

    def fast_hist(self, pred, gt, nclass):
        mask = (gt >= 0) & (gt < nclass)
        hist = np.bincount(
            nclass * gt[mask].astype(int) +
            pred[mask], minlength=nclass ** 2).reshape(nclass, nclass)
        return hist

    def fscore(self, prec, rec, f):
        '''
        f: 
            f0.5_score: f == 0.5
            f1_score: f == 1
            f2_score: f == 2
        '''
        return (((1+f**2) * prec) * rec) / (f**2 * prec + rec)

    def macroscore(self, hist):
        prec, rec, f1, f0_5 = [], [], [], []
        for i in range(len(hist)):
            prec.append(hist[i,i]/np.sum(hist[:,i]))
            rec.append(hist[i,i]/np.sum(hist[i,:]))
            f1.append(self.fscore(prec[i], rec[i], 1))
            f0_5.append(self.fscore(prec[i], rec[i], 0.5))
        return f1, f0_5, prec, rec

    def microscore(self, hist):
        prec = rec = np.sum(np.diag(hist)) / np.sum(hist)
        f1 = self.fscore(prec, rec, 1)
        f0_5 = self.fscore(prec, rec, 0.5)
        return f1, f0_5, prec, rec


if __name__ == '__main__':
    label_map = {80:1}
    # label_map = {10:1, 20:2, 60:3, 80:4}
    label_dir = rf"/data/data/landset30/choose_data/label/all"
    pred_dir = rf"/data/data/landset30/choose_data/clip_pred/clip_AWEIsh_5600open"
    eva_csvpath = r'/data/data/landset30/newunion/csv/water/clip_AWEIsh_5600open.csv'

    calimg_score(label_dir, pred_dir, label_map, eva_csvpath)

    