# encoding: utf-8
import os
import os.path as osp
import shutil
from common import check_dir, open_json
from process.preprocess_align import WindowClip
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

    # test_orimg_dir = rf"/data/dataset/change_detection/origin_merge/2016/label"
    # test_shiftul_path = rf"/data/data/change_detection/merge/256_128/2016/shiftul.json"
    # pred_dir = rf"/data/data/change_detection/models/drit_concat/unet/effb1_dicebce_resume/mask_update_1"
    # predmerge_dir = rf"/data/data/change_detection/models/drit_concat/unet/effb1_dicebce_resume/mask_update_1_bigmap"

    # check_dir(predmerge_dir)
    # shift_ul = open_json(test_shiftul_path)   # '0.jpg': ['tif_name', shift_x, shift_y]
    # windowc.merge(test_orimg_dir, pred_dir, predmerge_dir, shift_ul, cfg)   # merge pred


    # rename pred_merge
    # origindir = r'/data/dataset/semi_compete/origin/val'
    # csp_dict = corresponding_dict(origindir)
    # submission_dir = r'/data/data/semi_compete/submission/swintransform_weightcediceloss'
    # check_dir(submission_dir)
    # rename_copy(predmerge_dir, submission_dir, csp_dict)



    # merge other methods patch 
    # names= ['hm', 'reinhard', 'unit', 'train', 'cyclegan', 'drit']
    names= ['cyclegan']

    # test_orimg_dir = rf"/data/dataset/change_detection/origin_merge/2016/label"
    # test_shiftul_path = rf"/data/data/change_detection/merge/256_128/2016/shiftul.json"
    test_orimg_dir = rf"/data/dataset/update/test/mask"
    test_shiftul_path = rf"/data/data/update/256_128/test/shiftul.json"
    # test_orimg_dir = rf"/data/dataset/update/train/mask_modify"
    # test_shiftul_path = rf"/data/data/update/256_128/train_modify/shiftul.json"

    shift_ul = open_json(test_shiftul_path)   # '0.jpg': ['tif_name', shift_x, shift_y]

    # only one
    pred_dir = rf"/data/data/update/models/correct/unet/reinhard/pred_ep47"
    predmerge_dir = rf"/data/data/update/models/correct/unet/reinhard/pred_ep47_bigmap"
    check_dir(predmerge_dir)
    windowc.merge(test_orimg_dir, pred_dir, predmerge_dir, shift_ul, cfg)   # merge pred




    # metd_names = ['ocrnet', 'pspnet', 'segformer', 'swintransformer', 'deeplabv3', 'unet']
    # backb_names = ['hr18_dicebce', 'effb1_dicebce', 'b2_dicebce', 'upernet_swin-s_dicebce', 'effb1_dicebce', 'resnet50_dicebce']
    metd_names = ['unet']
    backb_names = ['effb3_dicebce_scse_size160']
    threds = [0, 0.2, 0.4, 0.6, 0.8, 1]

    # for name in names:
    #     for i, metd in enumerate(metd_names):
    #         for thred in threds:
    #         # pred_dir = rf"/data/data/update/models/{name}/{metd}/{backb_names[i]}/pred"
    #         # predmerge_dir = rf"/data/data/update/models/{name}/{metd}/{backb_names[i]}/pred_bigmap"
    #         # check_dir(predmerge_dir)
    #         # windowc.merge(test_orimg_dir, pred_dir, predmerge_dir, shift_ul, cfg)   # merge pred
    #             pred_dir = rf"/data/data/update/models/{name}/{metd}/{backb_names[i]}/mask_update_modify/mask_update_{thred}"
    #             predmerge_dir = rf"/data/data/update/models/{name}/{metd}/{backb_names[i]}/mask_update_modify/mask_update_{thred}_bigmap"
    #             check_dir(predmerge_dir)
    #             windowc.merge(test_orimg_dir, pred_dir, predmerge_dir, shift_ul, cfg)   # merge pred


    # merge different threds
    # names= ['0', '02', '04', '06', '08', '1']
    # names= ['image_cyclegan', 'image_drit_concat_resume', 'image_unit', 'image_reinhard', 'image_hm']
    # test_orimg_dir = rf"/data/dataset/change_detection/origin_merge/2016/label"
    # test_shiftul_path = rf"/data/data/change_detection/merge/256_128/2016/shiftul.json"
    # # test_orimg_dir = rf"/data/dataset/update/test/mask"
    # # test_shiftul_path = rf"/data/data/update/256_128/test/shiftul.json"
    # shift_ul = open_json(test_shiftul_path)   # '0.jpg': ['tif_name', shift_x, shift_y]
    # for name in names:
    #     pred_dir = rf"/data/data/change_detection/merge/256_128/2012/{name}"
    #     predmerge_dir = rf"/data/data/change_detection/merge/256_128/2012/trans_bigmap/{name}"
    #     check_dir(predmerge_dir)
    #     windowc.merge(test_orimg_dir, pred_dir, predmerge_dir, shift_ul, cfg, band=3)   # merge pred



    