import numpy as np
import pandas as pd
import os.path as osp
import cv2
import os
from tqdm import tqdm


def is_dir(dir):
    if not os.path.exists(dir): os.makedirs(dir)


def graytorgb(img):
    label_mapping_rgb = {0:(0,0,0),
                         1:(250,100,255),
                         2:(0,200,0),
                         3:(0, 100, 255),
                         4:(100,100,50),
                         5:(255, 0, 0),
                         6:(190,190,190),
                         7:(200,240,255),
                         8:(0,200,255),}
    img_rgb = np.stack(np.vectorize(label_mapping_rgb.get)(img),  axis=2).astype('uint8')
    return img_rgb


def dir2rgb(imgdir, rgbdir, names):
    for name in tqdm(names, total=len(names)):
        name = name[:-4] + '.png'
        imgpath = osp.join(imgdir, name)
        rgbpath = osp.join(rgbdir, name)
        img = cv2.imread(imgpath, 0)
        imgrgb = graytorgb(img)
        cv2.imwrite(rgbpath, imgrgb)


if __name__ == '__main__':
    tv_path = r'/data/data/multiclass/trainval.csv'
    # imgdir = r'/data/data/multiclass/train/labels'   # label
    # imgrgb_dir = r'/data/data/multiclass/val_rgb'
    predir = r'/data/data/multiclass/models/segformer/mit_b0_ce/results'   # pred
    predrgb_dir = r'/data/data/multiclass/models/segformer/mit_b0_ce/results_rgb'
    is_dir(predrgb_dir)

    # val name
    tvids = pd.read_csv(tv_path, dtype={'id': int})
    val_ids = tvids[tvids['id']!=0]['name'].tolist()

    prednames = os.listdir(predir)
    dir2rgb(predir, predrgb_dir, prednames)