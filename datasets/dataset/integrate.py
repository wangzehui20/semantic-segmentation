import os.path as osp
import os
import shutil
from tqdm import tqdm


def integrate(rootdir, dstdir):
    files = os.listdir(rootdir)
    for file in tqdm(files, total=len(files)):
        curdir = osp.join(rootdir, file+'/BDORTHO')
        names = os.listdir(curdir)
        for name in names:
            curpath = osp.join(curdir, name)
            shutil.copy(curpath, dstdir)
        

if __name__ == '__main__':
    labeled_path = r'/data/dataset/semi_compete/origin/labeled_train'
    unlabeled_path = r'/data/dataset/semi_compete/origin/unlabeled_train'
    val_path = r'/data/dataset/semi_compete/origin/val'

    dst_labeled_path = '/data/dataset/semi_compete/origin_integrate/labeled_train/image'
    dst_unlabeled_path = '/data/dataset/semi_compete/origin_integrate/unlabeled_train/image'
    dst_val_path = '/data/dataset/semi_compete/origin_integrate/val/image'

    integrate(labeled_path, dst_labeled_path)



