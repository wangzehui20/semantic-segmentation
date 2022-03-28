"""
writed by wangzehui in 2022.03.21
"""

import os
import numpy as np
import os.path as osp
import cv2
from tqdm import tqdm

def match(img, img_ref):
    _, _, c = img.shape
    img_out = np.zeros_like(img)
    for i in range(c):
        img_hist, _ = np.histogram(img[:,:,i], 256)     # 计算图像直方图
        ref_hist, _ = np.histogram(img_ref[:,:,i], 256)
        img_cdf = np.cumsum(img_hist)     # 计算图像累积分布函数
        ref_cdf = np.cumsum(ref_hist)
        for j in range(256):
            tmp = abs(img_cdf[j] - ref_cdf)
            tmp = tmp.tolist()
            index = tmp.index(min(tmp))     # 寻找tmp的最小值下标
            img_out[:,:,i][img[:,:,i]==j] = index  
    return img_out

def hist_match(oridir, refdir, dstdir):
    names = os.listdir(oridir)
    for name in tqdm(names, total=len(names)):
        oripath = osp.join(oridir, name)
        ref_name = name.replace('2019', '2018')
        refpath = osp.join(refdir, ref_name)
        dstpath = osp.join(dstdir, name)
        img = cv2.imread(oripath)
        img_ref = cv2.imread(refpath)
        img_matched = match(img, img_ref)
        cv2.imwrite(dstpath, img_matched)

def check_dir(dir):
    if not os.path.exists(dir): os.makedirs(dir)

if __name__ == '__main__':
    oridir = r'/data/data/update/256_128/train/image_gan'
    refdir = r'/data/data/update/256_128/test/image_gan'
    dstdir = r'/data/data/update/256_128/test/image_histmatch'
    check_dir(dstdir)
    hist_match(refdir, oridir, dstdir)
