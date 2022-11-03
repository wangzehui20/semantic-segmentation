import os
import os.path as osp
import cv2
import csv
import numpy as np
from collections import Counter
from tqdm import tqdm


def get_topk(img_dir, label_dir, topk, label_index):
    n_pixels = []
    img_paths = []
    label_paths = []
    label_list = os.listdir(label_dir)
    for name in tqdm(label_list, total=len(label_list)):
        label_path = osp.join(label_dir, name)
        img_path = osp.join(img_dir, name)
        label = cv2.imread(label_path, 0)
        if not np.any(label==label_index): continue
        counter = Counter(label.ravel())
        n_pixels.append(counter[label_index])
        img_paths.append(img_path)
        label_paths.append(label_path)
    idx = sorted(enumerate(n_pixels), key=lambda x: x[1])[::-1]
    topk_idx = idx[:topk]
    return [img_paths[i[0]] for i in topk_idx], [label_paths[i[0]] for i in topk_idx], topk_idx


def writecsv(logpath, data):
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


def record_ratio(csvpath, label_paths, topk_idx):
    for i in range(len(topk_idx)):
        data = dict()
        data['ratio'] = topk_idx[i][1] / (WIDTH * HEIGHT)
        data['label_path'] = osp.join(label_paths[i])
        writecsv(csvpath, data)


def record_meanration(mratio_csvpath, topk, topk_idx):
    meanratio_dict = dict()
    for cls in topk:
        meanratio_dict[cls] = 0

    # mean ratio of topk
    ratio = [(topk_idx[i][1] / (WIDTH * HEIGHT)) for i in range(len(topk_idx))]  
    for tk in topk:
        meanratio_dict[tk] = np.mean(ratio[:topk])
    writecsv(mratio_csvpath, meanratio_dict)


if __name__ == '__main__':
    WIDTH = 512
    HEIGHT = 512
    LABEL = 80

    # single class dir
    imgdir = rf"/data/data/landset30/Unet_bifpn/new512_128/train/image"
    labeldir = rf"/data/data/landset30/Unet_bifpn/new512_128/train/mask"
    ratio_csvpath = r'/data/data/landset30/Unet_bifpn/new512_128/building_pixratio.csv'
    mratio_csvpath = r'/data/data/landset30/Unet_bifpn/new512_128/building_topkmratio.csv'

    topk = [2000, 4000, 6000, 8000, 10000, len(os.listdir(imgdir))]
    img_paths, label_paths, topk_idx = get_topk(imgdir, labeldir, topk[-1], label_index=LABEL)
    print("Record topk image and label done")

    record_ratio(ratio_csvpath, label_paths, topk_idx)
    print("Record topk pixel ratio done")

    record_meanration(mratio_csvpath, topk, topk_idx)
    print("Record topk mean pixel ratio done")

    