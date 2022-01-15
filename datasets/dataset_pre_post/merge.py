# encoding: utf-8
from common import is_dir, open_json
from preprocess_align import WindowClip
from config import Config


if __name__ == '__main__':
    cfg = Config()
    windowc = WindowClip()

    cfg.RES_BASEDIR = rf"{cfg.RES_DIR}/newmodels_building/UnetPlusPlus/effb3_dicebce"
    test_orimg_dir = rf"{cfg.ORI_DIR}/test_image"
    test_shiftul_path = rf"{cfg.CLIP_BASEDIR}/test_shiftul.json"
    pred_dir = rf"{cfg.RES_BASEDIR}/pred"
    predmerge_dir = rf"/data/data/landset30/choose_data/clip_pred/clip_AWEIsh_5600open"

    is_dir(predmerge_dir)
    shift_ul = open_json(test_shiftul_path)   # '0.jpg': ['tif_name', shift_x, shift_y]
    windowc.merge(test_orimg_dir, pred_dir, predmerge_dir, shift_ul, cfg)   # merge pred


    