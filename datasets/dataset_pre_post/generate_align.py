# encoding: utf-8
import os
from tqdm import tqdm
from process.preprocess_align import data_process
from common import check_dir, get_imlist, save_json
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
            shiftul["{}_{}.tif".format(name[:-4], start_idx + j)] = (name, clipbox[2], clipbox[0])   # name remove '.tif'
        start_idx = end_idx
    return shiftul, statis_dict


if __name__ == '__main__':
    cfg = Config()

    imori_dir = rf"{cfg.ORI_DIR}/image"
    maskori_dir = rf"{cfg.ORI_DIR}/mask_modify"
    imclp_dir = rf"{cfg.CLIP_DIR}/{cfg.FILE_NAME}/train_modify/image"
    maskclp_dir = rf"{cfg.CLIP_DIR}/{cfg.FILE_NAME}/train_modify/mask"
    statis_path = rf"{cfg.CLIP_DIR}/{cfg.FILE_NAME}/train_modify/statis.json"
    shiftul_path = rf"{cfg.CLIP_DIR}/{cfg.FILE_NAME}/train_modify/shiftul.json"
    check_dir(imclp_dir)
    check_dir(maskclp_dir)
    check_dir(os.path.dirname(statis_path))
    check_dir(os.path.dirname(shiftul_path))

    shiftul, statis_dict = startest(imori_dir, imclp_dir, cfg, 
                                        maskori_dir=maskori_dir, maskdst_dir=maskclp_dir)
    print(f"Generate image successfully")

    save_json(statis_path, statis_dict)
    print(f"Generate statistics successfully")

    save_json(shiftul_path, shiftul)
    print(f"Generate shift_ul json successfully")
    