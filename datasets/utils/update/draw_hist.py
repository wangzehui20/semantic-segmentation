import os
import os.path as osp
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import MultipleLocator

def draw_one_band_hist(hist, save_dir, color, rng = 0.03):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置加载的字体名
    plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题
    plt.rcParams['axes.facecolor'] = 'white'
    y_major_locator = MultipleLocator(0.001) # 设置坐标轴刻度
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.figure(figsize=(10,3))
    plt.bar(range(len(hist)), hist, fc=color, width=1.1)
    plt.ylim((0,rng)) # 设置坐标轴范围
    for i in np.arange(0, rng, 0.01):
        plt.axhline(y=i, c='black', lw=0.8) # 设置与坐标轴平行的线
    plt.xticks(fontsize=12) # 设置坐标字体大小
    plt.yticks(np.arange(0,rng+0.01,0.01),fontsize=12)
    plt.savefig(osp.join(save_dir, "{}.png".format(color)))
    plt.close()

def draw_hist(hist, save_dir, colors):
    fig = plt.figure(figsize=(80,40))
    plt.rcParams['axes.facecolor'] = 'white'
    for i in range(len(hist)):
        plt.subplot(1,3,i+1)
        plt.bar(range(len(hist[i])), hist[i], fc=colors[i], width=1.1)
        plt.ylim((0,5))
    plt.savefig(osp.join(save_dir, "bgr_hist.png"))

def cal_one_band_hist(im):
    # one band
    h, w = im.shape
    counter = Counter(im.flatten().tolist())
    keys_set = set(counter.keys())
    hist = []
    for i in range(256):
        if i not in keys_set:
            hist.append(0)
        else:
            hist.append(counter[i] / (h*w))   # y unit is 1e-2
    return hist

def cal_hist(im):
    """
    return: [[b], [g], [r]]
    """
    # three band
    c = im.shape[2]
    hist = []
    for i in range(c):
        hist.append(cal_one_band_hist(im[:,:,i]))
    return np.array(hist)

def cal_hist_dir(im_dir, save_dir, rng=0.03):
    colors = ['deepskyblue', 'chartreuse', 'orangered']
    im_names = os.listdir(im_dir)
    total_hist = np.zeros((3, 256))
    for name in im_names:
        im_path = osp.join(im_dir, name)
        im = cv2.imread(im_path)
        hist = cal_hist(im)
        total_hist += hist
    total_hist /= len(im_names)
    for i, color in enumerate(colors):
        draw_one_band_hist(total_hist[i], save_dir, color, rng)
    # draw_hist(total_hist, save_dir, colors)

def merge_im(save_dir):
    gap = 60
    im_merge_path = osp.join(save_dir, 'rgb.png')
    im_b = cv2.imread(osp.join(save_dir, 'deepskyblue.png'))
    im_g = cv2.imread(osp.join(save_dir, 'chartreuse.png'))
    im_r = cv2.imread(osp.join(save_dir, 'orangered.png'))
    h, w, c = im_b.shape
    im_merge = np.zeros((h, w*3-6*gap, c))
    im_merge[:,:w-2*gap,:] = im_r[:,gap:w-gap,:]
    im_merge[:,w-2*gap:2*(w-2*gap),:] = im_g[:,gap:w-gap,:]
    im_merge[:,2*(w-2*gap):3*(w-2*gap),:] = im_b[:,gap:w-gap,:]
    cv2.imwrite(im_merge_path, im_merge)

def copy_image(im_dir, im_dst_dir):
    names = ['image_cyclegan', 'image_drit', 'image_unit', 'image_reinhard', 'image_hm']
    ano_names = ['image_reinhard', 'image_hm']
    for name in names:
        im_dst_base_dir = osp.join(im_dst_dir, name)
        check_dir(im_dst_base_dir)
        im_path = osp.join(im_dir, name, '2018_1638.png')
        im_dst_path = osp.join(im_dst_dir, name, '2018_1638.png')
        shutil.copyfile(im_path, im_dst_path)

        cal_hist_dir(im_dst_base_dir, im_dst_base_dir)
        merge_im(im_dst_base_dir)

def check_dir(dir):
    if not os.path.exists(dir): os.makedirs(dir)


if __name__ == '__main__':
    # rng = 0.03
    # im_dir = r'/data/data/change_detection/merge/256_128/2012/hist/image_drit_concat_resume'
    # cal_hist_dir(im_dir, im_dir, rng)
    # merge_im(im_dir)


    im_dir = r'/data/data/update/256_128/train'
    im_dst_dir = r'/data/data/update/256_128/train/dispaly'
    copy_image(im_dir, im_dst_dir)