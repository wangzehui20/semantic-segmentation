import os.path as osp
import numpy as np
import cv2
import os
from common import get_mean_std, is_lowimg
from tqdm import tqdm
from multiprocessing import Pool, Manager
from .window_clip import WindowClip
from .utils import TifInfo
from .label_align import LabelAlign, CoordinateTransform


def data_process(imori_path, imdst_dir, cfg, maskori_dir=None, maskdst_dir=None, start=0, lock=None, mode='test', is_align=False, bg_filter=False):
    # 1 读取栅格数据
    imori_info = TifInfo(imori_path)
    imname = imori_path.split('/')[-1]
    typeTf = TypeTransform()
    imori_coordT = CoordinateTransform(imori_info.dt)
    windowc = WindowClip()
    if is_align:
        lblA = LabelAlign()
        freq = None  # 本次搜索优先考虑上一次的搜索结果

    # 2 影像分块
    cliplist = windowc.get_cliplist(imori_info.w, imori_info.h, cfg.WIDTH, cfg.HEIGHT, cfg.OVERLAP)
    statis = []

    # 3 根据裁剪框对训练数据和真值裁剪
    for i, clip in tqdm(enumerate(cliplist), total=len(cliplist)):
        y_min, y_max, x_min, x_max = clip[0], clip[1], clip[2], clip[3]
        im_w = clip[3] - clip[2]
        im_h = clip[1] - clip[0]
        # start from upper-left point
        corner_xy = [(x_min, y_min),(x_max, y_min),(x_max, y_max),(x_min, y_max)]   # [(),()...]
        corner_geo = []
        for xy in corner_xy:
            corner_geo.append(imori_coordT.xy2geo(xy))

        # lock
        if lock:
            lock.acquire()
        
        if maskori_dir and is_align:
            maskrange = lblA.get_maskrange(maskori_dir)
            maskname = lblA.find_label(imori_coordT, maskori_dir, maskrange, corner_geo, freq)

            # 没有在真值图像里找到
            if mode == 'train' and maskname is None:
                continue
        else:
            maskname = imori_path.split('/')[-1]

        # (1) 存储裁剪的训练图片
        imdst_path = osp.join(imdst_dir, "{}_{}.png".format(imname[:-4], start))
        imdata = imori_info.dt.ReadAsArray(x_min, y_min, im_w, im_h)  # 获取分块数据

        # image nodata is 256, ---spetial
        imdata[imdata==256] = 0
        
        if im_w != cfg.WIDTH or im_h != cfg.HEIGHT:
            imdata_pad = np.zeros((imori_info.b, cfg.HEIGHT, cfg.WIDTH)).astype(typeTf.gdal2np(imori_info.t))
            imdata_pad[:, :im_h, :im_w] = imdata[:, :im_h, :im_w]
            imdata = imdata_pad

        # 背景像素过多则不保存该图片
        if mode == 'train' and bg_filter and maskori_dir and is_lowimg(imdata):
            continue

        # 16位转8位
        if imori_info.t == 'UInt16' and np.any(imdata>255):
            imdata = typeTf.uint16Touint8(imdata)

        # 保存为带坐标的.tif
        # save_tif(imori_path, imdata, imdst_path, corner_geo[0])   #.tif
        # cv2.imwrite(imdst_path, imdata.transpose(1,2,0)[:,:,::-1].astype(np.uint8))    #.png

        # test_gt, need to save all images
        if mode == 'test' and maskname is None:
            continue

        # (2) 存储裁剪的真值图片
        if maskori_dir:

            maskdst_path = osp.join(maskdst_dir, "{}_{}.png".format(imname[:-4], start))
            maskori_path = osp.join(maskori_dir, imname)
            maskori_info = TifInfo(maskori_path)
            maskori_coordT = CoordinateTransform(maskori_info.dt)
            (valid_w, valid_h) = windowc.valid_hw(imori_info, corner_xy[0], cfg)

            maskcorner_geo = []
            maskcorner_xy = []
            for c in corner_geo:
                maskc_geo = imori_coordT.coordsys(maskori_info.dt, c) if is_align else c
                maskcorner_geo.append(maskc_geo)
                
                maskc_xy = maskori_coordT.geo2xy(maskc_geo)
                maskcorner_xy.append(maskc_xy)
            if is_align and lblA.is_transf(imori_info.pj, maskori_info.pj):
                
                # 计算最小外接矩形
                rectul_xy, rect_w, rect_h = lblA.getminrect(maskcorner_xy)
                rectbl_xy = (rectul_xy[0], rectul_xy[1]+rect_h)
                rect_mat = maskori_info.dt.ReadAsArray(rectul_xy[0], rectul_xy[1], rect_w, rect_h)

                # calculate angle
                basevec = np.array(maskcorner_xy[3])-np.array(maskcorner_xy[0])
                rotvec = np.array(rectbl_xy)-np.array(rectul_xy)
                angle = lblA.cal_angle(basevec, rotvec)

                # 旋转后的对应真值矩阵
                rotrect_mat = lblA.rotate_bound(rect_mat, angle)
                center = (rotrect_mat.shape[0]/2, rotrect_mat.shape[1]/2)
                clpmat_ul = np.array((round(center[0]-valid_h/2)-1, round(center[1]-valid_w/2)-1))

                if clpmat_ul[0]<0: clpmat_ul[0]=0
                if clpmat_ul[1]<0: clpmat_ul[1]=0

                maskdata = rotrect_mat[clpmat_ul[0]:clpmat_ul[0]+valid_h,clpmat_ul[1]:clpmat_ul[1]+valid_w]
            else:
                # mask比image少一个像素
                if y_min>=17408 and 'huairou' in imname:    # ---spetial
                    maskdata = maskori_info.dt.ReadAsArray(maskcorner_xy[0][0], maskcorner_xy[0][1]-1, valid_w, valid_h)
                elif x_min>=21120 and 'shunyi' in imname:   # ---spetial
                    maskdata = maskori_info.dt.ReadAsArray(maskcorner_xy[0][0]-1, maskcorner_xy[0][1], valid_w, valid_h)
                else:
                    maskdata = maskori_info.dt.ReadAsArray(maskcorner_xy[0][0], maskcorner_xy[0][1], valid_w, valid_h)
            
            # mask nodata is 255, ---spetial
            maskdata[maskdata==255] = 0
            
            # 单波段则增加band维度
            if len(maskdata.shape) == 2:
                maskdata = maskdata[np.newaxis, :, :]

            if (valid_w < cfg.WIDTH) or (valid_h < cfg.HEIGHT):
                maskdata_pad = np.zeros((maskori_info.b, cfg.HEIGHT, cfg.WIDTH)).astype(typeTf.gdal2np(maskori_info.t))
                maskdata_pad[:, :valid_h, :valid_w] = maskdata[:, :valid_h, :valid_w]
                maskdata = maskdata_pad

            # 若真值背景像素过多则移除之前保存的训练图像
            if mode=='train' and bg_filter and maskori_dir and is_lowimg(maskdata):
                os.remove(imdst_path)
                continue
            
            # remove image background but has label
            maskdata = mask_imgbg(imdata, maskdata)

            # 将裁剪的真值数据保存到裁剪的训练图片坐标系中
            # save_tif(imori_path, maskdata, maskdst_path, corner_geo[0])   #.tif

            # 测试集裁剪把mask全为0的图片筛选出来, ---spetial
            # if np.all(imdata==0) or np.all(maskdata==0) or is_lowimg(imdata) or is_lowimg(maskdata):  
            if np.all(imdata==0):
                maskdst_path = maskdst_path[:-4] + '_bg.png'
                imdst_path = imdst_path[:-4] + '_bg.png'
            cv2.imwrite(imdst_path, imdata.transpose(1, 2, 0)[:, :, ::-1].astype(np.uint8))  # .png
            cv2.imwrite(maskdst_path, convert_label(maskdata.squeeze(), inverse=True))  # .png

            if lock:
                lock.release()

        # 保存其均值方差信息
        statis.append(get_mean_std(np.transpose(imdata, (1, 2, 0))))
        start += 1
    return start, cliplist, np.mean(statis, axis=0).tolist()


def convert_label(label, inverse=False):
    label_mapping = {
        0: 0,
        255: 1
    }
    tmp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[tmp == k] = v
    else:
        for k, v in label_mapping.items():
            label[tmp == k] = v
        label[label > len(label_mapping) - 1] = 0
    return label


def data_process_multi(inputs):
    pool = Pool()
    manager = Manager()
    lock=manager.Lock()
    zip_inputs = list(zip(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]), lock)
    results = pool.starmap(data_process, zip_inputs)
    pool.close()
    pool.join()
    return results


def mask_imgbg(img, label):
    """
    filter no image but label
    """
    msk = np.ones(label.shape).astype(np.bool)
    for i in range(len(img)):
        msk = (msk * img[i]).astype(np.bool)
    label = (label * msk)
    return label


class TypeTransform():
    def __init__(self) -> None:
        pass

    def uint16Touint8(self, img):
        """
        img: (band, h, w)
        """
        band = len(img)
        assert band != 1 or band != 3 
        for i in range(len(img)):
            im = img[i,:,:]
            if np.any(im!=0):
                maxm = max(im[im!=0])
                minm = min(im[im!=0])
                im[im!=0] = ((im[im!=0]-minm) / (maxm-minm) * 255).astype(np.uint8)
            img[i,:,:] = im
            im[np.isnan(im)] = 0
        return img

    def gdal2np(self, gdaltype):
        if gdaltype == 'UInt8':
            return np.uint8
        elif gdaltype == 'UInt16':
            return np.uint16
        else:
            return np.float32