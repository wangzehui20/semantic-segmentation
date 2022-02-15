import numpy as np
import os
import os.path as osp
import cv2
import pandas as pd
from tqdm import tqdm
from common import check_dir


def graytorgb(img):
    label_mapping_rgb = {0: (34, 31, 32),
                         1: (204, 102, 92),
                         2: (209, 154, 98),
                         3: (217, 208, 106),
                         4: (182, 218, 106),
                         5: (142, 217, 105),
                         6: (140, 194, 130),
                         7: (111, 175, 98),
                         8: (219, 245, 215),
                         9: (186, 255, 180),
                         10: (55, 126, 34),
                         11: (111, 174, 167),
                         12: (145, 96, 38),
                         13: (103, 153, 214),
                         14: (41, 96, 246),
                         15: (34, 31, 32),
                         }
    img_rgb = np.stack(np.vectorize(label_mapping_rgb.get)(img), axis=2).astype('uint8')
    return img_rgb


def start(rootdir, dstdir, names):
    for name in tqdm(names, total=len(names)):
        path = osp.join(rootdir, name[:-4]+'.png')
        dstpath = osp.join(dstdir, name[:-4]+'.png')
        img = cv2.imread(path, 0)
        img_rgb = graytorgb(img)
        cv2.imwrite(dstpath, img_rgb)


if __name__ == '__main__':
    # rootdir = r'/data/data/semi_compete/clip_integrate/1024_384/labeled_train/mask'
    # dstdir = r'/data/data/semi_compete/clip_integrate/1024_384/labeled_train/val_mask_rgb'
    # ids_csv = r'/data/data/semi_compete/clip_integrate/1024_384/labeled_train/val.csv'

    # 512
    # rootdir = r'/data/data/semi_compete/clip_integrate/512_128/labeled_train/mask'
    # dstdir = r'/data/data/semi_compete/clip_integrate/512_128/labeled_train/val_mask_rgb'
    # ids_csv = r'/data/data/semi_compete/clip_integrate/512_128/labeled_train/val.csv'

# -----------------
    # pred
    rootdir = r'/data/data/semi_compete/models/ocrnet/hr18/pred_val'
    dstdir = r'/data/data/semi_compete/models/ocrnet/hr18/pred_val_rgb'
    ids_csv = r'/data/data/semi_compete/clip_integrate/512_128/labeled_train/val.csv'

    check_dir(dstdir)
    names = pd.read_csv(ids_csv)['name'].tolist() if ids_csv and osp.exists(ids_csv) else None
    start(rootdir, dstdir, names)
    
