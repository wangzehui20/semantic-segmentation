import os
import cv2
import numpy as np
from tqdm import tqdm
from .utils import get_imlist, TifInfo, save_tif

class WindowClip():
    def __init__(self) -> None:
        pass
    
    # 滑动窗口的形式返回裁剪区域
    def get_cliplist(self, width, height, clipw, cliph, overlap, direction=False):
        """
        direction: False represents x axis and True represents y axis 
        """
        start_w = 0
        start_h = 0
        end_w = clipw
        end_h = cliph
        crop_box_list = []
        if not direction:
            while start_h < height:
                if end_h > height:
                    end_h = height
                while start_w < width:
                    if end_w > width:
                        end_w = width
                    crop_box_list.append([start_h, end_h, start_w, end_w])
                    if end_w == width: break
                    start_w = end_w - overlap
                    end_w = start_w + clipw
                if end_h == height: break
                start_h = end_h - overlap
                end_h = start_h + cliph
                start_w = 0
                end_w = clipw
        else:
            while start_w < width:
                if end_w > width:
                    end_w = width
                while start_h < height:
                    if end_h > height:
                        end_h = height
                    crop_box_list.append([start_h, end_h, start_w, end_w])
                    if end_h == height: break
                    start_h = end_h - overlap
                    end_h = start_h + cliph
                if end_w == width: break
                start_w = end_w - overlap
                end_w = start_w + clipw
                start_h = 0
                end_h = cliph
        return crop_box_list

    def valid_hw(self, info, xy, cfg):
        valid_w = info.w - xy[0] if xy[0] + cfg.WIDTH > info.w else cfg.WIDTH
        valid_h = info.h - xy[1] if xy[1] + cfg.HEIGHT > info.h else cfg.HEIGHT
        return (int(valid_w), int(valid_h))

    def get_valid_size(self, info, clipul_xy, cfg):
        valid_width = info.width - clipul_xy[0] if clipul_xy[0] + cfg.WIDTH > info.width else cfg.WIDTH
        valid_height = info.height - clipul_xy[1] if clipul_xy[1] + cfg.HEIGHT > info.height else cfg.HEIGHT
        return (int(valid_width), int(valid_height))

    # half overlap merge
    def recover_clip_box(self, pred_dir, mask_merge, img_info, cfg):
        n_band = len(mask_merge)
        img_path = os.path.join(pred_dir, img_info[0][:-4]+'.png')   # normal
        # img_path = os.path.join(pred_dir, img_info[0][:-4]+'.png').replace('2016_', '2012_')   # tmp, train data split
        # img_path = os.path.join(pred_dir, img_info[0][:-4]+'.png').replace('2019_', '2018_')   # tmp, train data split, update dataset
        if n_band > 1:
            img = cv2.imread(img_path).transpose(2,0,1)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = img[np.newaxis,:,:]
        if img is None:
            img = np.zeros((n_band, cfg.HEIGHT, cfg.WIDTH))
        # pred
        # img_path = os.path.join(pred_dir, img_info[0][:-4]+'.npy')
        # img = np.load(img_path)
        shift_x, shift_y = img_info[1], img_info[2]
        half_overlap = int(cfg.OVERLAP / 2)
        # x orientation
        if shift_x == 0:
            # clip image
            start_x = 0
            end_x = cfg.WIDTH - half_overlap
            # mask_merge
            start_shift_x = shift_x + start_x
            end_shift_x = shift_x + end_x
            # y orientation
            if shift_y == 0:
                # clip image
                start_y = 0
                end_y = cfg.HEIGHT - half_overlap
                # mask_merge
                start_shift_y = shift_y + start_y
                end_shift_y = shift_y + end_y
            elif shift_y != 0:
                # clip image
                start_y = half_overlap
                end_y = cfg.HEIGHT - half_overlap
                # mask_merge
                start_shift_y = shift_y + start_y
                end_shift_y = shift_y + end_y

        elif shift_x != 0:
            # clip image
            start_x = half_overlap
            end_x = cfg.WIDTH - half_overlap
            # mask_merge
            start_shift_x = shift_x + start_x
            end_shift_x = shift_x + end_x
            # y orientation
            if shift_y == 0:
                # clip image
                start_y = 0
                end_y = cfg.HEIGHT - half_overlap
                # mask_merge
                start_shift_y = shift_y + start_y
                end_shift_y = shift_y + end_y
            elif shift_y != 0:
                # clip image
                start_y = half_overlap
                end_y = cfg.HEIGHT - half_overlap
                # mask_merge
                start_shift_y = shift_y + start_y
                end_shift_y = shift_y + end_y
        mask_merge[:, start_shift_y:end_shift_y, start_shift_x:end_shift_x] = img[:, start_y:end_y, start_x:end_x]
        return mask_merge

    # next image union last image
    def recover_clip_box_(self, pred_dir, mask_merge, img_info, cfg):
        img_path = os.path.join(pred_dir, img_info[0])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = mask_merge.shape
        shift_x, shift_y = img_info[1], img_info[2]

        for i in range(0, cfg.HEIGHT):
            for j in range(0, cfg.WIDTH):
                cur_shift_y = shift_y+i
                cur_shift_x = shift_x+j
                if cur_shift_y>h or cur_shift_x>w:
                    continue
                if img[i, j]:
                    mask_merge[cur_shift_y, cur_shift_x] = img[i, j]
        return mask_merge

    def get_bigtif2imgs(self, shift_ul):
        """
        大图裁剪为小图，记录小图的左上角坐标
        """
        tif2img_list = {}
        for key, value in shift_ul.items():
            tif_name = value[0]
            if tif_name not in tif2img_list.keys(): tif2img_list[tif_name] = []   # 'tif_name': ['0.jpg', shift_x, shift_y]
            img_info = [key, value[1], value[2]]
            tif2img_list[tif_name].append(img_info)
        return tif2img_list


    # merge from small clip image to full image
    def merge(self, orimg_dir, pred_dir, predmerge_dir, shift_ul, cfg, band=1):
        tif_list = get_imlist(orimg_dir)
        tif2img_list = self.get_bigtif2imgs(shift_ul)

        for name in tqdm(tif_list, total=len(tif_list)):
            orimg_info = TifInfo(os.path.join(orimg_dir, name))
            (height, width) = orimg_info.h, orimg_info.w
            height_extend = height + cfg.HEIGHT - cfg.OVERLAP
            width_extend = width + cfg.WIDTH - cfg.OVERLAP
            mask_merge = np.zeros((band, height_extend, width_extend))

            for img_info in tif2img_list[name]:
                mask_merge = self.recover_clip_box(pred_dir, mask_merge, img_info, cfg)
                # mask_merge = recover_clip_box_(pred_dir, mask_merge, img_info, cfg)   # union
            mask_merge = mask_merge[:, :height, :width]

            ul_lonlat = (orimg_info.tf[0], orimg_info.tf[3])
            save_tif(os.path.join(orimg_dir, name), mask_merge, os.path.join(predmerge_dir, name), ul_lonlat)
            # cv2.imwrite(os.path.join(seg_merge_dir, "{}.png".format(tif.split('.')[0])), mask_merge)   # .png