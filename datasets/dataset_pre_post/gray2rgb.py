import numpy as np
import os
import os.path as osp
import cv2
from tqdm import tqdm


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


def start(rootdir, dstdir):
    names = os.listdir(rootdir)
    for name in tqdm(names, total=len(names)):
        path = osp.join(rootdir, name)
        dstpath = osp.join(dstdir, name[:-4]+'.png')
        img = cv2.imread(path, 0)
        img_rgb = graytorgb(img)
        cv2.imwrite(dstpath, img_rgb)


if __name__ == '__main__':
    rootdir = r'/data/dataset/semi_compete/origin_integrate/labeled_train/mask'
    dstdir = r'/data/dataset/semi_compete/origin_integrate/labeled_train/mask_rgb'
    start(rootdir, dstdir)
    
