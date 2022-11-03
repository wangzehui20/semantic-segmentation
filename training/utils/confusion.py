import os
import os.path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def confusion(gtdir, predir, labels):
    gtlist = os.listdir(gtdir)
    confus = np.zeros((len(labels), len(labels)))
    for gt in tqdm(gtlist, total=len(gtlist)):
        gtpath = osp.join(gtdir, gt)
        predpath = osp.join(predir, gt)
        gt = cv2.imread(gtpath, 0).reshape(-1)
        pred = cv2.imread(predpath, 0).reshape(-1)
        confus += confusion_matrix(gt, pred, labels=labels)
    return confus


def plot_confusion(confus, classes):
    confus_path = r'/data/data/landset30/multiclass/confusion.png'
    plt.figure()
    plt.imshow(confus, cmap=plt.cm.Blues)

    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False

    plt.ylabel('gt')
    plt.xlabel('pred')
    plt.title('Confusion matrix')

    indices = range(len(confus))
    plt.xticks(indices, classes, rotation=45)    #设置横坐标方向，rotation=45为45度倾斜
    plt.yticks(indices, classes)

    for i in range(len(confus)):
        for j in range(len(confus)):
            plt.text(j, i, confus[i][j], #注意是将confusion[i][j]放到j,i这个位置
                 fontsize=8,
                 horizontalalignment="center",  # 水平居中
                 verticalalignment="center",    # 垂直居中
                 color="white" if confus[i, j] > confus.max()/2. else "black") #颜色控制

    plt.savefig(confus_path)


if __name__ == '__main__':
    labels = [0, 10, 20, 30, 60, 80]
    classes = ['background', 'farmland', 'forest', ' grass', 'water', ' surface']

    labelmerge_dir = r'/data/data/landset30/Unet_bifpn_building/512_128/mergelabel'
    pred_multi = r'/data/data/landset30/multiclass/test_22_0818'
    
    confus = confusion(labelmerge_dir, pred_multi, labels)
    plot_confusion(confus, classes)


