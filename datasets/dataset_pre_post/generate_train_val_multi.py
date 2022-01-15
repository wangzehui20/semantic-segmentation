# encoding: utf-8
import os
from preprocess_align import get_maskrange, data_process
from common import is_dir, get_imlist, save_json
from config import Config
from multiprocessing import Pool


def get_inputs(orimg_dir, dstimg_dir, orilabel_dir, dstlabel_dir, label_range, cfg):
    tif_list = get_imlist(orimg_dir)

    orimgs_path = [os.path.join(orimg_dir, tif) for tif in tif_list]
    dstimgs_dir = [dstimg_dir for i in range(len(tif_list))]
    cfgs = [cfg for i in range(len(tif_list))]
    orilabels_dir = [orilabel_dir for i in range(len(tif_list))]
    dstlabels_dir = [dstlabel_dir for i in range(len(tif_list))]
    start_idxs = [i * cfg.IMAGE_STEP for i in range(len(tif_list))]
    labels_range = [label_range for i in range(len(tif_list))]

    inputs = [orimgs_path, dstimgs_dir, cfgs, orilabels_dir, dstlabels_dir, start_idxs, labels_range]
    return inputs


# multi process
def get_train_data(orimg_dir, dstimg_dir, orilabel_dir, dstlabel_dir, label_range, cfg):
    idxs = []
    statis_list = []

    inputs = get_inputs(orimg_dir, dstimg_dir, orilabel_dir, dstlabel_dir, label_range, cfg)
    results = multi_data_process(inputs)
    for i in range(len(results)):
        statis_list.append(results[i][2])   # results[i]: start, clip_list, statistics
        idxs.extend([i * cfg.IMAGE_STEP + j for j in range(len(results[i][2]))])
    return idxs, statis_list


def multi_data_process(inputs):
    zip_inputs = list(zip(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]))
    pool = Pool()
    results = pool.starmap(data_process, zip_inputs)
    pool.close()
    pool.join()
    return results


if __name__ == '__main__':
    cfg = Config()
    train_orimg_dir = rf"{cfg.ORI_DIR}/train_image"
    train_orilabel_dir = rf"{cfg.ORI_DIR}/train_label"
    train_clpimg_dir = rf"{cfg.CLIP_BASEDIR}/train/image"
    train_clplabel_dir = rf"{cfg.CLIP_BASEDIR}/train/mask"
    train_labelrange_path = rf"{cfg.CLIP_BASEDIR}/range.json"
    train_statis_path = rf"{cfg.CLIP_BASEDIR}/train_statis.json"
    is_dir(train_clpimg_dir)
    is_dir(train_clplabel_dir)

    # record label range to generate train-val data
    label_range = get_maskrange(train_orilabel_dir)
    save_json(train_labelrange_path, label_range)
    print("Generate train label range successfully")

    idxs, statis_list = get_train_data(train_orimg_dir, train_clpimg_dir, train_orilabel_dir, train_clplabel_dir,
                                       label_range, cfg)
    print("Generate train image and label successfully")

    save_json(train_statis_path, statis_list)
    print("Generate train image statistics successfully")
