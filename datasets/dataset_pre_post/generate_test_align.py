# encoding: utf-8
import os
from tqdm import tqdm
from preprocess_align import data_process, LabelAlign
from common import is_dir, get_imlist, save_json
from config import Config


def startest(imori_dir, imdst_dir, cfg, maskori_dir=None, maskdst_dir=None):
    imlist = get_imlist(imori_dir)
    start_idx = 0
    statis_dict = {}
    shiftul = {}
    for name in tqdm(imlist, total=len(imlist)):
        impath = os.path.join(imori_dir, name)
        end_idx, cliplist, statis = data_process(impath, imdst_dir, cfg, maskori_dir=maskori_dir,
                                                    maskdst_dir=maskdst_dir, start=start_idx, mode='test')
        statis_dict[name] = statis
        for j, clipbox in enumerate(cliplist):
            # (imname, upper-left x, upper-left y)
            shiftul["{}_{}.tif".format(name, start_idx + j)] = (name, clipbox[2], clipbox[0])
        start_idx = end_idx
    return shiftul, statis_dict


def folder():
    cfg = Config()
    imori_root_path = rf'{cfg.ORI_DIR}/unlabeled_train'
    names = os.listdir(imori_root_path)
    for name in names:
        imori_dir = rf"{cfg.ORI_DIR}/unlabeled_train/{name}/BDORTHO"
        maskori_dir = rf"{cfg.ORI_DIR}/unlabeled_train/{name}/UrbanAtlas"
        imclp_dir = rf"{cfg.CLIP_DIR}/unlabeled_train/{cfg.FILE_NAME}/{name}/image"
        maskclp_dir = rf"{cfg.CLIP_DIR}/unlabeled_train/{cfg.FILE_NAME}/{name}/mask"
        statis_path = rf"{cfg.CLIP_DIR}/unlabeled_train/{cfg.FILE_NAME}/{name}/test_statis.json"
        shiftul_path = rf"{cfg.CLIP_DIR}/unlabeled_train/{cfg.FILE_NAME}/{name}/test_shiftul.json"
        is_dir(imclp_dir)
        is_dir(maskclp_dir)
        is_dir(os.path.dirname(statis_path))
        is_dir(os.path.dirname(shiftul_path))

        shiftul, statis_dict = startest(imori_dir, imclp_dir, cfg, 
                                            maskori_dir=None, maskdst_dir=None)
        print(f"Generate test {name} image successfully")

        save_json(statis_path, statis_dict)
        print(f"Generate test {name} statistics successfully")

        save_json(shiftul_path, shiftul)
        print(f"Generate test {name} shift_ul json successfully")


if __name__ == '__main__':
    folder()
    