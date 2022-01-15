# encoding: utf-8
import os
from tqdm import tqdm
from preprocess_align import LabelAlign, data_process
from common import is_dir, get_imlist, save_json
from config import Config


def get_train_data(orimg_dir, dstimg_dir, orilabel_dir, dstlabel_dir, cfg):
    tif_list = get_imlist(orimg_dir)
    start_idx = 0
    statis_dict = {}
    # 先将裁剪的所有图片放至train文件夹，再将所属val的image从train文件夹移至val文件夹
    for tif in tqdm(tif_list, total=len(tif_list)):
        tif_path = os.path.join(orimg_dir, tif)
        start_idx, _, statis = data_process(tif_path, dstimg_dir, cfg, maskori_dir=orilabel_dir,
                                                         maskdst_dir=dstlabel_dir, \
                                                         start=start_idx, mode='train')
        statis_dict[tif] = statis
    return start_idx, statis_dict


def folder():
    cfg = Config()
    train_orimg_root_dir = rf'{cfg.ORI_DIR}/labeled_train'
    names = os.listdir(train_orimg_root_dir)
    for name in names:
        train_orimg_dir = rf"{cfg.ORI_DIR}/labeled_train/{name}/BDORTHO"
        train_orilabel_dir = rf"{cfg.ORI_DIR}/labeled_train/{name}/UrbanAtlas"
        train_clpimg_dir = rf"{cfg.CLIP_DIR}/labeled_train/{cfg.FILE_NAME}/{name}/image"
        train_clplabel_dir = rf"{cfg.CLIP_DIR}/labeled_train/{cfg.FILE_NAME}/{name}/mask"
        train_statis_path = rf"{cfg.CLIP_DIR}/labeled_train/{cfg.FILE_NAME}/{name}/train_statis.json"
        is_dir(train_clpimg_dir)
        is_dir(train_clplabel_dir)

        end_idx, statis_dict = get_train_data(train_orimg_dir, train_clpimg_dir, train_orilabel_dir, train_clplabel_dir, cfg)
        print(f"Generate train {name} image and label successfully")

        save_json(train_statis_path, statis_dict)
        print(f"Generate train {name} image statistics successfully")


if __name__ == '__main__':
    folder()    
