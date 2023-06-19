import cv2
import numpy as np
import shapely
import os
import os.path as osp
from shapely.geometry import Polygon
from tqdm import tqdm

def check_dir(dir):
    if not os.path.exists(dir): os.makedirs(dir)

def mask2cntrs(mask):
    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # 出现area为空的情况
    if len(contours) == 0: return None
    contours_filter = []
    for j, contour in enumerate(contours):
        if len(contour)>=3:
            contours_filter.append(np.squeeze(contour))   # (count_num, 1, 2)
    return contours_filter

def cntr2poly(contour):
    return Polygon(contour).convex_hull

def get_cntrs(pred, mask_ref, iou_thred=0):
    pred_cntrs = mask2cntrs(pred)
    mask_ref_cntrs = mask2cntrs(mask_ref)
    cntrs = []
    if not pred_cntrs:
        return cntrs
    if not mask_ref_cntrs:
        return pred_cntrs

    for pred_cntr in pred_cntrs:
        poly = cntr2poly(pred_cntr)
        update_flag = False
        for mask_ref_cntr in mask_ref_cntrs:
            poly_ref = cntr2poly(mask_ref_cntr)
            # iou = cal_iou_sub(poly, poly_ref, poly_ref_flag=False)
            iou_ref = cal_iou_sub(poly, poly_ref)
            if iou_ref > iou_thred:
                # if iou > iou_ref:
                #     break
                cntrs.append(mask_ref_cntr)
                update_flag = True
                # break
        if not update_flag:
            # pred_cntr = cv2.approxPolyDP(pred_cntr)
            cntrs.append(pred_cntr)
    return cntrs

def cal_iou_sub(poly, poly_ref, poly_ref_flag=True):
    if not poly.intersects(poly_ref):
        return 0
    else:
        try:
            inter_area = poly.intersection(poly_ref).area
            union_area = poly_ref.area if poly_ref_flag else poly.area
            iou_sub = float(inter_area/union_area)
            return iou_sub
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            return 0

def update(pred_path, mask_ref_path, mask_update_path, im_size=256, iou_thred=0):
    pred = cv2.imread(pred_path, 0)
    mask_ref = cv2.imread(mask_ref_path, 0)
    cntrs = get_cntrs(pred, mask_ref, iou_thred)
    mask_update = np.zeros((im_size, im_size))
    for cntr in cntrs:
        cv2.fillPoly(mask_update, cntr[np.newaxis,:, :], 255)
    cv2.imwrite(mask_update_path, mask_update)

def update_dir(pred_dir, mask_ref_dir, mask_update_dir, iou_thred=0):
    """
    replace prediction to before mask when both are intersect most
    :param pred_dir:
    :param mask_ref_dir:
    :param mask_update_dir:
    :return:
    """
    pred_names = os.listdir(pred_dir)
    for name in tqdm(pred_names, total=len(pred_names)):
        pred_path = osp.join(pred_dir, name)
        # mask_ref_path = osp.join(mask_ref_dir, name.replace('2016_merge', '2012_merge'))   # replace name
        mask_ref_path = osp.join(mask_ref_dir, name.replace('2019_', '2018_'))   # replace name
        mask_update_path = osp.join(mask_update_dir, name)
        update(pred_path, mask_ref_path, mask_update_path, iou_thred=iou_thred)

####################################
# merge pred and mask_ref directly #
####################################

def overlap(pred_path, mask_ref_path, mask_update_path):
    label_map = {0:0, 255:1}
    pred = cv2.imread(pred_path, 0)
    mask_ref = cv2.imread(mask_ref_path, 0)
    for k, v in label_map.items():
        pred[pred == k] = v
        mask_ref[mask_ref == k] = v
    mask_update = mask_ref + pred
    mask_update[mask_update>0] = 255
    cv2.imwrite(mask_update_path, mask_update)

def overlap_dir(pred_dir, mask_ref_dir, mask_update_dir):
    """
    replace prediction to before mask when both are intersect most
    :param pred_dir:
    :param mask_ref_dir:
    :param mask_update_dir:
    :return:
    """
    pred_names = os.listdir(pred_dir)
    for name in tqdm(pred_names, total=len(pred_names)):
        pred_path = osp.join(pred_dir, name)
        mask_ref_path = osp.join(mask_ref_dir, name.replace('2016_merge', '2012_merge'))   # replace name
        # mask_ref_path = osp.join(mask_ref_dir, name.replace('2019_', '2018_'))   # replace name
        mask_update_path = osp.join(mask_update_dir, name)
        overlap(pred_path, mask_ref_path, mask_update_path)

###### ###
#  demo  #
## #######

def demo_dir(mask_dir, mask_ref_dir, pred_dir, mask_update_dir, im_demo_dir, im_dir=None, im_ref_dir=None):
    """
    display before and after image and mask, pred, update result
    :param mask_dir:
    :param mask_ref_dir:
    :param pred_dir:
    :param mask_update_dir:
    :param im_demo_dir:
    :param im_dir:
    :param im_ref_dir:
    :return:
    """
    pred_names = os.listdir(pred_dir)
    for name in tqdm(pred_names, total=len(pred_names)):
        mask_path = osp.join(mask_dir, name)
        pred_path = osp.join(pred_dir, name)
        # mask_ref_path = osp.join(mask_ref_dir, name.replace('2016_merge', '2012_merge'))  # replace name
        mask_ref_path = osp.join(mask_ref_dir, name.replace('2019_', '2018_'))  # replace name
        mask_update_path = osp.join(mask_update_dir, name)
        im_path = osp.join(im_dir, name)
        # im_ref_path = osp.join(im_ref_dir, name.replace('2016_merge', '2012_merge'))
        im_ref_path = osp.join(im_ref_dir, name.replace('2019_', '2018_'))

        # demo
        if im_demo_dir:
            im_demo_path = osp.join(im_demo_dir, name)
            demo(mask_path, pred_path, mask_ref_path, mask_update_path, im_demo_path, im_path, im_ref_path)

def demo(mask_path, mask_ref_path, pred_path, mask_update_path, im_demo_path, im_path, im_ref_path):
    mask = cv2.imread(mask_path, 0)
    pred = cv2.imread(pred_path, 0)
    mask_ref = cv2.imread(mask_ref_path, 0)
    mask_update = cv2.imread(mask_update_path, 0)
    im = cv2.imread(im_path)
    im_ref = cv2.imread(im_ref_path)
    im_demo = np.zeros((pred.shape[0], pred.shape[0]*6, 3))
    im_demo[:, :256, :] = im_ref
    im_demo[:, 256:512, :] = np.expand_dims(mask_ref, 2).repeat(3, axis=2)
    im_demo[:, 512:768, :] = im
    im_demo[:, 768:1024, :] = np.expand_dims(mask, 2).repeat(3, axis=2)
    im_demo[:, 1024:1280, :] = np.expand_dims(pred, 2).repeat(3, axis=2)
    im_demo[:, 1280:1536, :] = np.expand_dims(mask_update, 2).repeat(3, axis=2)
    cv2.imwrite(im_demo_path, im_demo)

##############
# merge mask #
##############

def merge_part_mask(mask_update_path, pred_part_path, mask_merge_path):
    label_map = {0:0, 255:1}
    mask_update = cv2.imread(mask_update_path)
    mask_part = cv2.imread(pred_part_path)
    for k, v in label_map.items():
        mask_update[mask_update == k] = v
        mask_part[mask_part == k] = v
    mask_merge = mask_update + mask_part
    mask_merge[mask_merge>0] = 255
    cv2.imwrite(mask_merge_path, mask_merge)

def merge_part_mask_dir(mask_update_dir, pred_part_dir, mask_merge_dir):
    """
    merge update result and part prediction
    :param mask_update_dir: update result dir
    :param pred_part_dir: part prediction dir
    :param mask_merge_dir: reupdate dir
    :return:
    """
    names = os.listdir(mask_update_dir)
    for name in tqdm(names, total=len(names)):
        mask_update_path = osp.join(mask_update_dir, name)
        pred_part_path = osp.join(pred_part_dir, name)
        mask_merge_path = osp.join(mask_merge_dir, name)
        merge_part_mask(mask_update_path, pred_part_path, mask_merge_path)


if __name__ == '__main__':
    name = 'cyclegan'
    # pred_dir = rf'/data/data/change_detection/models/{name}/unet/effb1_dicebce/pred'
    # mask_dir = r'/data/data/change_detection/merge/256_128/2016/mask'
    # mask_ref_dir = r'/data/data/change_detection/merge/256_128/2012/mask'
    # mask_update_dir = rf'/data/data/change_detection/models/{name}/unet/effb1_dicebce/mask_update_PL/'
    # im_demo_dir = r'/data/data/change_detection/models/cyclegan/unet/effb1_dicebce/im_demo'
    # pred_part_dir = r'/data/data/change_detection/models/cyclegan/unet/effb1_dicebce/pred_part'
    # mask_merge_dir = r'/data/data/change_detection/models/cyclegan/unet/effb1_dicebce/mask_merge_1'

    # pred_dir = r'/data/data/update/models/cyclegan/unet/effb1_dicebce/pred'
    # mask_dir = r'/data/data/update/256_128/test/mask'
    # mask_ref_dir = r'/data/data/update/256_128/train/mask'
    # mask_update_dir = r'/data/data/update/models/cyclegan/unet/effb1_dicebce/mask_overlap'
    # im_demo_dir = r'/data/data/update/models/cyclegan/unet/effb1_dicebce/im_demo'
    # pred_part_dir = r'/data/data/update/models/cyclegan/unet/effb1_dicebce/pred_part'
    # mask_merge_dir = r'/data/data/update/models/cyclegan/unet/effb1_dicebce/mask_merge_0'

    # check_dir(mask_update_dir)
    # check_dir(im_demo_dir)
    # check_dir(mask_merge_dir)
    # im_dir = r'/data/data/change_detection/merge/256_128/2016/image'   # after image
    # im_ref_dir = r'/data/data/change_detection/merge/256_128/2012/image'   # before image

    # im_dir = r'/data/data/update/256_128/test/image_filter'   # after image
    # im_ref_dir = r'/data/data/update/256_128/train/image_filter'   # before image


    # update_dir(pred_dir, mask_ref_dir, mask_update_dir)
    # # demo_dir(mask_dir, mask_ref_dir, pred_dir, mask_update_dir, im_demo_dir, im_dir, im_ref_dir)
    # merge_part_mask_dir(mask_update_dir, pred_part_dir, mask_merge_dir)

    # overlap_dir(pred_dir, mask_ref_dir, mask_update_dir)



    # only update for other methods
    # names= ['hm', 'reinhard', 'unit', 'drit', 'train']
    # mask_ref_dir = r'/data/data/update/256_128/train/mask'
    # for name in names:
    #     pred_dir = rf'/data/data/update/models/{name}/unet/effb1_dicebce/pred'
    #     mask_update_dir = rf'/data/data/update/models/{name}/unet/effb1_dicebce/mask_update_0'
    #     check_dir(mask_update_dir)
    #     update_dir(pred_dir, mask_ref_dir, mask_update_dir)


    # different threds
    pred_dir = rf'/data/data/update/models/{name}/unet/effb3_dicebce/pred'
    # mask_ref_dir = r'/data/data/update/256_128/train_modify/mask'
    mask_ref_dir = r'/data/data/update/256_128/train/mask'
    # mask_ref_dir = r'/data/data/change_detection/merge/256_128/2012/mask'
    threds = [0]
    for thred in threds:
        # mask_update_dir = rf'/data/data/change_detection/models/{name}/unet/effb3_dicebce_scse_edge/mask_update_modify/mask_update_{thred}'
        mask_update_dir = rf'/data/data/update/models/{name}/unet/effb3_dicebce/mask_update/mask_update_{thred}'
        check_dir(mask_update_dir)
        update_dir(pred_dir, mask_ref_dir, mask_update_dir, iou_thred=thred)


    



        

