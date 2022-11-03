import os
import os.path as osp
import h5py
import numpy as np
import cv2
import csv
from tqdm import tqdm

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
            elif isinstance(v, list):
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

def h52img(h5dir, imgdir):
    h5_names = os.listdir(h5dir)
    mean_list = []
    std_list = []
    meanstd_csv_path = r'/data/code/semantic-segmentation-semi-supervised-learning/config/huapo/meanstd.csv'
    c = Csv()
    for h5_name in tqdm(h5_names, total=len(h5_names)):
        h5path = osp.join(h5dir, h5_name)
        imgpath = osp.join(imgdir, h5_name.split('_')[1][:-3] + '.png')
        data = get_data(h5path).transpose(2,0,1)[1:4] # bgr
        img = truncated_linear_stretch(data)
        cv2.imwrite(imgpath, img.transpose(1,2,0))

        # mean and std
        # data = data.reshape(3,-1)
        # mean_list.append(np.mean(data, axis=1).tolist())
        # std_list.append(np.std(data, axis=1).tolist())
    # d = dict()
    # d['mean'] = np.mean(mean_list, 0).tolist()
    # d['std'] = np.mean(std_list, 0).tolist()
    # c.writecsv(meanstd_csv_path, d)

def h52mask(h5dir, maskdir):
    h5_names = os.listdir(h5dir)
    for h5_name in tqdm(h5_names, total=len(h5_names)):
        h5path = osp.join(h5dir, h5_name)
        maskpath = osp.join(maskdir, h5_name.split('_')[1][:-3] + '.png')
        mask = get_data(h5path)
        cv2.imwrite(maskpath, convert_label(mask, inverse=True))

def get_data(path):
        # flag: 0 is image, 1 is mask
        f = h5py.File(path, 'r')
        data = f['img'][:]
        f.close()
        return data

def truncated_linear_stretch(image, truncated_value=1, max_out = 255, min_out = 0):
    def gray_process(gray):
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out 
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        if(max_out <= 255):
            gray = np.uint8(gray)
        elif(max_out <= 65535):
            gray = np.uint16(gray)
        return gray
    
    #  如果是多波段
    if(len(image.shape) == 3):
        image_stretch = []
        for i in range(image.shape[0]):
            gray = gray_process(image[i])
            image_stretch.append(gray)
        image_stretch = np.array(image_stretch)
    #  如果是单波段
    else:
        image_stretch = gray_process(image)
    return image_stretch

def convert_label(label, inverse=False):
        label_mapping = {
            0: 0,
            255: 1,
        }
        tmp = label.copy()
        if inverse:
            for v, k in label_mapping.items():
                label[tmp == k] = v
        else:
            for k, v in label_mapping.items():
                label[tmp == k] = v
            label[label > len(label_mapping) - 1] = 0
        return label

if __name__ == '__main__':
    h5dir = r'/data/data/huapo/ValidData/img'
    imgdir = r'/data/data/huapo/ValidData_/img'
    h52img(h5dir, imgdir)

    # h5dir = r'/data/data/huapo/TrainData/mask'
    # maskdir = r'/data/data/huapo/TrainData_/mask'
    # h52mask(h5dir, maskdir)
