"""
writed by wangzehui in 2022.03.29
"""

import os
import numpy as np
import os.path as osp
import cv2
from tqdm import tqdm

def match(img, img_ref):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2LAB)
    img_means = [0, 0, 0]
    img_ref_means = [0, 0, 0]
    img_stdevs = [0, 0, 0]
    img_ref_stdevs = [0, 0, 0]
    lab_img = lab_img.astype(np.float) / 255
    lab_img_ref = lab_img_ref.astype(np.float) / 255
    for i in range(3):
        img_means[i] += lab_img[:,:,i].mean()
        img_ref_means[i] += lab_img_ref[:,:,i].mean()
        img_stdevs[i] += lab_img[:,:,i].std()
        img_ref_stdevs[i] += lab_img_ref[:,:,i].std()
    thred = [a / b for a, b in zip(img_stdevs, img_ref_stdevs)]
    lab_img_out = thred * (lab_img - img_means) + img_ref_means
    
    lab_img_out = lab_img_out * 255
    lab_img_out *= (lab_img_out > 0)
    lab_img_out = (lab_img_out * (lab_img_out <= 255) + 255 * (lab_img_out > 255)).astype(np.uint8)
    img_out = cv2.cvtColor(lab_img_out, cv2.COLOR_LAB2BGR)
    return img_out

def reinhard_match(oridir, refdir, dstdir):
    names = os.listdir(oridir)
    for name in tqdm(names, total=len(names)):
        oripath = osp.join(oridir, name)
        ref_name = name.replace('2012', '2016')
        refpath = osp.join(refdir, ref_name)
        dstpath = osp.join(dstdir, name)
        img = cv2.imread(oripath)
        img_ref = cv2.imread(refpath)
        img_matched = match(img, img_ref)
        cv2.imwrite(dstpath, img_matched)

def check_dir(dir):
    if not os.path.exists(dir): os.makedirs(dir)

if __name__ == '__main__':
    oridir = r'/data/data/change_detection/merge/256_128/2012/image'
    refdir = r'/data/data/change_detection/merge/256_128/2016/image'
    dstdir = r'/data/data/change_detection/merge/256_128/2012/image_reinhard'
    check_dir(dstdir)
    reinhard_match(oridir, refdir, dstdir)
