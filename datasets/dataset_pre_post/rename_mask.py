import os
import os.path as osp
import shutil
from tabnanny import check
from tqdm import tqdm
from common import check_dir


def corresponding_dict(dir):
    csp = dict()
    names = os.listdir(dir)
    for name in tqdm(names, total=len(names)):
        subdir = osp.join(dir, name, 'BDORTHO')
        csp[name] = []
        subnames = os.listdir(subdir)
        for sname in tqdm(subnames, total=len(subnames)):
            csp[name].append(sname)
    return csp

def rename_copy(dir, dstdir, csp_dict):
    for key in csp_dict.keys():
        check_dir(osp.join(dstdir, key))

    names = os.listdir(dir)
    for name in tqdm(names, total=len(names)):
        path = osp.join(dir, name)
        for key, value in csp_dict.items():
            if name not in value:
                continue
            else:
                dstname = name[:-4] + '_prediction.tif'
                dstpath = osp.join(dstdir, key, dstname)
                shutil.copyfile(path, dstpath)


if __name__ == '__main__':
    rootdir = r'/data/dataset/semi_compete/origin/val'
    csp_dict = corresponding_dict(rootdir)

    predir = r'/data/data/semi_compete/models_pseudo/segformer/b0/pred_merge'
    submission_dir = r'/data/data/semi_compete/submission/segformer_b0_pseudo'
    check_dir(submission_dir)
    rename_copy(predir, submission_dir, csp_dict)
