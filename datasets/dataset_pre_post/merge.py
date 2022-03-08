# encoding: utf-8
import os
import os.path as osp
import shutil
from common import check_dir, open_json
from preprocess_align import WindowClip
from config import Config
from common import check_dir
from tqdm import tqdm


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
    cfg = Config()
    windowc = WindowClip()

    cfg.RES_BASEDIR = rf"{cfg.RES_DIR}/models/swinTransform/upernet_swin-s_weightcediceloss"
    test_orimg_dir = rf"{cfg.ORI_DIR}/val/image"
    test_shiftul_path = rf"{cfg.CLIP_DIR}/512_128/val/test_shiftul.json"
    pred_dir = rf"{cfg.RES_BASEDIR}/pred"
    predmerge_dir = rf"{cfg.RES_BASEDIR}/pred_merge"

    check_dir(predmerge_dir)
    shift_ul = open_json(test_shiftul_path)   # '0.jpg': ['tif_name', shift_x, shift_y]
    windowc.merge(test_orimg_dir, pred_dir, predmerge_dir, shift_ul, cfg)   # merge pred


    # rename pred_merge
    origindir = r'/data/dataset/semi_compete/origin/val'
    csp_dict = corresponding_dict(origindir)
    submission_dir = r'/data/data/semi_compete/submission/swintransform_weightcediceloss'
    check_dir(submission_dir)
    rename_copy(predmerge_dir, submission_dir, csp_dict)



    