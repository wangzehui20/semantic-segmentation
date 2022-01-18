# encoding: utf-8
from common import check_dir, open_json
from preprocess_align import WindowClip
from config import Config


if __name__ == '__main__':
    cfg = Config()
    windowc = WindowClip()

    cfg.RES_BASEDIR = rf"{cfg.RES_DIR}/models/segformer/b0"
    test_orimg_dir = rf"{cfg.ORI_DIR}/val/image"
    test_shiftul_path = rf"{cfg.CLIP_DIR}/512_128/val/test_shiftul.json"
    pred_dir = rf"{cfg.RES_BASEDIR}/pred"
    predmerge_dir = rf"{cfg.RES_BASEDIR}/pred_merge"

    check_dir(predmerge_dir)
    shift_ul = open_json(test_shiftul_path)   # '0.jpg': ['tif_name', shift_x, shift_y]
    windowc.merge(test_orimg_dir, pred_dir, predmerge_dir, shift_ul, cfg)   # merge pred


    