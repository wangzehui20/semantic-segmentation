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
    labelist = get_imlist(predmerge_dir)
    hist = np.zeros((classes, classes))
    e = Evaluate()
    c = Csv()
    for name in tqdm(labelist, total=len(labelist)):
        # labelpath = osp.join(labelmerge_dir, name.replace('2012_merge', '2016_merge'))
        # labelpath = osp.join(labelmerge_dir, name.replace('2016_merge', '2012_merge'))   # tmp, train data split
        # labelpath = osp.join(labelmerge_dir, name.replace('2018_', '2019_'))
        # labelpath = osp.join(labelmerge_dir, name.replace('2019', '2018'))   # tmp, train data split, update dataset
        labelpath = osp.join(labelmerge_dir, name)   # tmp, train data split, update dataset
        predpath = osp.join(predmerge_dir, name)
        # label = read_image2(labelpath)
        # pred = read_image2(predpath)
        label = cv2.imread(labelpath, 0)
        pred = cv2.imread(predpath, 0)
        label[label==255] = 0   # nodata value, update dataset origin mask
        pred[label==255] = 0   # nodata value, update dataset origin mask
        # label = label[:, 25856:]   # tmp, train data split
        # pred = pred[:, 25856:]   # tmp, train data split
        # label[:5632, :256] = 0   # tmp, train data split
        # label = label[12800:, :]   # tmp, train data split, update dataset
        # pred = pred[12800:, :]   # tmp, train data split
        # label[:256, :10624] = 0   # tmp, train data split
        # label = label[:8814, :]   # tmp, train data ft
        # pred = pred[:8814, :]   # tmp, train data ft
        h, w = label.shape
        pred = pred[:h, :w]

        for k, v in label_map.items():
            label[label==k] = v
            pred[pred==k] = v
        label[label>len(label_map)] = 0
        pred[pred>len(label_map)] = 0
        hist += e.fast_hist(pred, label, classes)
    f1, f0_5, precision, recall = e.macroscore(hist)
    # print(f1, f0_5, precision, recall)
    # hist = np.array([[2.1549734e8, 2.37636e6], [8.82478e5, 3.64128e7]])
    iou = e.iou(hist)
    precision = e.precision(hist)
    recall = e.recall(hist)
    oa = e.oa(hist)
    f1 = e.f1(hist)
    # print(hist)
    print(' iou:', iou, ' precision:', precision, ' recall:', recall, ' oa:', oa, ' f1:', f1)

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
    hist: TN FP
          FN TP
    """
    def __init__(self) -> None:
        self.eps = 1e-8
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

    def iou(self, hist):
        intersection = hist[-1,-1]
        union = np.sum(hist[-1,:]) + np.sum(hist[:,-1]) - intersection + self.eps
        iou = intersection / union
        return iou

    def oa(self, hist):
        oa = (hist[0,0] + hist[1,1]) / (np.sum(hist) + self.eps)
        return oa

    def precision(self, hist):
        precision = hist[1,1] / (hist[0,1] + hist[1,1] + self.eps)
        return precision

    def recall(self, hist):
        recall = hist[1,1] / (hist[1,0] + hist[1,1] + self.eps)
        return recall

    def f1(self, hist):
        return self.fscore(self.precision(hist), self.recall(hist), 1)

if __name__ == '__main__':
    label_map = {255:1}
    # label_map = {10:1, 20:2, 60:3, 80:4}
    # label_dir = rf"/data/dataset/change_detection/origin_merge/2016/label"
    # pred_dir = rf"/data/data/change_detection/models/cyclegan/unet/effb1_dicebce_edge_only_sa/pred_bigmap"

    # label_dir = rf"/data/data/update/256_128/test/mask"
    # pred_dir = rf"/data/data/update/models/cyclegan/unet/effb1_dicebce/mask_overlap"

    eva_csvpath = r'/data/data/landset30/newunion/csv/water/clip_AWEIsh_5600open.csv'

    # calimg_score(label_dir, pred_dir, label_map, eva_csvpath)



    # names= ['hm', 'reinhard', 'unit', 'train', 'cyclegan', 'drit']
    names= ['cyclegan']

    label_dir = r'/data/dataset/update/test/mask'
    # label_dir = rf"/data/dataset/change_detection/origin_merge/2016/label"
    # label_dir = rf"/data/dataset/change_detection/origin_merge/2012/label"   # tmp, train data split
    # label_dir = rf"/data/dataset/update/train/mask"   # tmp, train data split, update dataset

    # only one
    pred_dir = rf'/data/data/update/models/correct/unet/reinhard/pred_ep47_bigmap'
    calimg_score(label_dir, pred_dir, label_map, eva_csvpath)


    # pred_dir = r'/data/dataset/update/train/mask_modify'
    # pred_dir = r'/data/dataset/change_detection/origin_merge/2012/label'
    # calimg_score(label_dir, pred_dir, label_map, eva_csvpath)


    # metd_names = ['ocrnet', 'pspnet', 'segformer', 'swintransformer', 'deeplabv3', 'unet']
    # backb_names = ['hr18_dicebce', 'effb1_dicebce', 'b2_dicebce', 'upernet_swin-s_dicebce', 'effb1_dicebce', 'resnet50_dicebce']
    metd_names = ['unet']
    backb_names = ['effb3_dicebce_scse_size160']
    threds = [0, 0.2, 0.4, 0.6, 0.8, 1]

    # for name in names:
    #     for i, metd in enumerate(metd_names):
    #         for thred in threds:
    #             pred_dir = rf'/data/data/update/models/{name}/{metd}/{backb_names[i]}/mask_update_modify/mask_update_{thred}_bigmap'
    #             calimg_score(label_dir, pred_dir, label_map, eva_csvpath)

            # pred_dir = rf'/data/data/update/models/{name}/{metd}/{backb_names[i]}/pred_bigmap'
            # calimg_score(label_dir, pred_dir, label_map, eva_csvpath)


    # evaluate different threds
    # names= ['0', '02', '04', '06', '08', '1']
    # # label_dir = rf"/data/dataset/change_detection/origin_merge/2016/label"
    # label_dir = r'/data/dataset/update/test/mask'
    # for name in names:
    #     pred_dir = rf"/data/data/update/models/cyclegan/unet/effb1_dicebce/mask_update_{name}_bigmap"
    #     calimg_score(label_dir, pred_dir, label_map, eva_csvpath)

    