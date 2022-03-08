import os.path as osp
from cv2 import imwrite
import numpy as np
import math
import cv2
import os
from pyproj import Transformer
from common import get_imlist,  get_mean_std, is_lowimg
from osgeo import gdal, osr
from tqdm import tqdm
from multiprocessing import Pool, Manager


def data_process(imori_path, imdst_dir, cfg, maskori_dir=None, maskdst_dir=None, start=0, lock=None, mode='test', is_align=False):
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
        corner_xy = [(x_min, y_min),(x_max, y_min),(x_max, y_max),(x_min, y_max)]   #[(),()...]
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
        if im_w != cfg.WIDTH or im_h != cfg.HEIGHT:
            imdata_pad = np.zeros((imori_info.b, cfg.HEIGHT, cfg.WIDTH)).astype(typeTf.gdal2np(imori_info.t))
            imdata_pad[:, :im_h, :im_w] = imdata[:, :im_h, :im_w]
            imdata = imdata_pad

        # 背景像素过多则删掉该图片
        # if mode == 'train' and maskori_dir and is_lowimg(imdata):
        #     continue

        # 16位转8位
        # imdata = typeTf.uint16Touint8(imdata)

        # 保存为带坐标的.tif
        # save_tif(imori_path, imdata, imdst_path, corner_geo[0])   #.tif
        cv2.imwrite(imdst_path, imdata.transpose(1,2,0)[:,:,::-1])    #.png

        # test_gt, need to save all images
        if mode == 'test' and maskname is None:
            continue

        # (2) 存储裁剪的真值图片
        if maskori_dir:

            maskdst_path = osp.join(maskdst_dir, "{}_{}.png".format(imname[:-4], start))
            maskori_path = osp.join(maskori_dir, imname)
            maskori_info = TifInfo(maskori_path)
            maskori_coordT = CoordinateTransform(maskori_info.dt)
            # label proj
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
                maskdata = maskori_info.dt.ReadAsArray(maskcorner_xy[0][0], maskcorner_xy[0][1], valid_w, valid_h)

            # 单波段则增加band维度
            if len(maskdata.shape) == 2:
                maskdata = maskdata[np.newaxis, :, :]

            if (valid_w < cfg.WIDTH) or (valid_h < cfg.HEIGHT):
                maskdata_pad = np.zeros((maskori_info.b, cfg.HEIGHT, cfg.WIDTH)).astype(typeTf.gdal2np(maskori_info.t))
                maskdata_pad[:, :valid_h, :valid_w] = maskdata[:, :valid_h, :valid_w]
                maskdata = maskdata_pad

            # 若真值背景像素过多则移除之前保存的训练图像
            # if maskdst_dir and is_lowimg(maskdata):
            #     os.remove(imdst_path)
            #     continue

            # remove image background but has label
            maskdata = mask_imgbg(imdata, maskdata)

            # 将裁剪的真值数据保存到裁剪的训练图片坐标系中
            # save_tif(imori_path, maskdata, maskdst_path, corner_geo[0])   #.tif
            cv2.imwrite(maskdst_path, convert_label(maskdata.squeeze(), inverse=True))  # .png

            if lock:
                lock.release()

        # 保存其均值方差信息
        statis.append(get_mean_std(np.transpose(imdata, (1, 2, 0))))
        start += 1
    return start, cliplist, np.mean(statis, axis=0).tolist()


class TifInfo():
    def __init__(self, path):
        self.dt = gdal.Open(path)
        self.w = self.dt.RasterXSize
        self.h = self.dt.RasterYSize
        self.b = self.dt.RasterCount
        self.t = gdal.GetDataTypeName(self.dt.GetRasterBand(1).DataType)
        self.tf = self.dt.GetGeoTransform()
        self.pj = self.dt.GetProjection()


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
    msk = np.ones(label.shape).astype(np.bool)
    for i in range(3):
        msk = (msk * img[i]).astype(np.bool)
    label = (label * msk)
    return label


def save_tif(orimg_path, img, dstimg_path, clipul_lonlat):
    '''
    保存裁剪图像为tif格式
    '''
    # data_type = gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType)
    # if data_type == 'Byte':
    #     data_type = gdal.GDT_Byte
    # elif data_type == 'UInt16':
    #     data_type = gdal.GDT_UInt16
    # else:
    #     data_type = gdal.GDT_Float32

    # stretch, default byte
    data_type = gdal.GDT_Byte

    img_band, img_height, img_width = img.shape
    dataset = gdal.Open(orimg_path)
    driver = gdal.GetDriverByName("GTiff")
    dst_dataset = driver.Create(dstimg_path, img_width, img_height, img_band, data_type)
    if dst_dataset:
        transf = list(dataset.GetGeoTransform())
        transf[0], transf[3] = clipul_lonlat[0], clipul_lonlat[1]
        transf = tuple(transf)
        dst_dataset.SetGeoTransform(transf)
        dst_dataset.SetProjection(dataset.GetProjection())

    for i in range(img_band):
        dst_dataset.GetRasterBand(i + 1).WriteArray(img[i])
    del dst_dataset


class LabelAlign():
    def __init__(self) -> None:
        pass

    # 判断裁图区域是否在真值的某一幅图像中
    def is_inside(self, imori_coordT, maskori_dir, maskrange, corner_geo, labelname):
        # 真值图的边界范围
        l_geo = maskrange[labelname][0][0]
        u_geo = maskrange[labelname][0][1]
        r_geo = maskrange[labelname][1][0]
        b_geo = maskrange[labelname][1][1]
        for c in corner_geo:
            # 裁图区域的坐标转换为真值的坐标
            dst_dataset = gdal.Open(osp.join(maskori_dir, labelname))
            dst_geo = imori_coordT.coordsys(dst_dataset, c)
            dstx_geo = dst_geo[0]
            dsty_geo = dst_geo[1]
            # 纬度的坐标方向是从下往上
            if not (dstx_geo < r_geo and dstx_geo > l_geo and dsty_geo > b_geo and dsty_geo < u_geo):
                return False
        return True

    # statistic origin label coordinate range
    def get_maskrange(self, maskdir):
        masklist = get_imlist(maskdir)
        maskrange = {}
        for name in masklist:
            maskpath = osp.join(maskdir, name)
            info = TifInfo(maskpath)
            coordT = CoordinateTransform(info.dt)
            maskrange[name] = coordT.xys2geos(np.array([[0, 0], [info.w, info.h]])).tolist()
        return maskrange

    # freq是上一次返回的label name
    def find_label(self, imori_coordT, maskori_dir, maskrange, corner_geo, freq=None):
        if freq and self.is_inside(imori_coordT, maskori_dir, maskrange, corner_geo, freq):
            return freq
        for labelname in maskrange.keys():
            if self.is_inside(imori_coordT, maskori_dir, maskrange, corner_geo, labelname):
                return labelname
        return None

    def cal_angle(self, basevec, rotvec):
        basevec_len = np.sqrt(basevec.dot(basevec))
        rotvec_len = np.sqrt(rotvec.dot(rotvec))
        cos_angle = basevec.dot(rotvec) /  (basevec_len*rotvec_len)
        angle = np.arccos(cos_angle)
        angle = angle*360/2/np.pi
        return angle

    def is_transf(self, orikwt, dstkwt):
        ori_EPSG = self.get_proj_transf(orikwt)
        dst_EPSG = self.get_proj_transf(dstkwt)
        return False if int(ori_EPSG) == int(dst_EPSG) else True

    def get_proj_transf(self, geos_kwt):
        proj = osr.SpatialReference(geos_kwt)
        space_EPSG = proj.GetAttrValue('AUTHORITY', 1)
        return space_EPSG

    def rotate_bound(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_NEAREST, borderValue=0)

    def getminrect(self, corner):
        xmax = ymax = -999999
        xmin = ymin = 999999
        for c in corner:
            if c[0]>xmax: xmax = c[0]
            if c[0]<xmin: xmin = c[0]
            if c[1]>ymax: ymax = c[1]
            if c[1]<ymin: ymin = c[1]
        ul = (xmin, ymin)
        width = math.ceil(xmax - xmin)
        height = math.ceil(ymax - ymin)
        return ul, width, height


class WindowClip():
    def __init__(self) -> None:
        pass
    
    # 滑动窗口的形式返回裁剪区域
    def get_cliplist(self, width, height, clipw, cliph, overlap):
        start_w = 0
        start_h = 0
        end_w = clipw
        end_h = cliph
        crop_box_list = []
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
        img_path = os.path.join(pred_dir, img_info[0][:-4]+'.png')
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
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

        mask_merge[start_shift_y:end_shift_y, start_shift_x:end_shift_x] = img[start_y:end_y, start_x:end_x]
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
    def merge(self, orimg_dir, pred_dir, predmerge_dir, shift_ul, cfg):
        tif_list = get_imlist(orimg_dir)
        tif2img_list = self.get_bigtif2imgs(shift_ul)

        for name in tqdm(tif_list, total=len(tif_list)):
            orimg_info = TifInfo(os.path.join(orimg_dir, name))
            (height, width) = orimg_info.h, orimg_info.w
            height_extend = height + cfg.HEIGHT - cfg.OVERLAP
            width_extend = width + cfg.WIDTH - cfg.OVERLAP
            mask_merge = np.zeros((height_extend, width_extend))

            for img_info in tif2img_list[name]:
                mask_merge = self.recover_clip_box(pred_dir, mask_merge, img_info, cfg)
                # mask_merge = recover_clip_box_(pred_dir, mask_merge, img_info, cfg)   # union
            mask_merge = mask_merge[:height, :width]
            mask_merge = mask_merge[np.newaxis,:,:]

            ul_lonlat = (orimg_info.tf[0], orimg_info.tf[3])
            save_tif(os.path.join(orimg_dir, name), mask_merge, os.path.join(predmerge_dir, name), ul_lonlat)
            # .png
            # cv2.imwrite(os.path.join(seg_merge_dir, "{}.png".format(tif.split('.')[0])), mask_merge)


class CoordinateTransform():
    def __init__(self, dataset=None) -> None:
        self.dataset = dataset

    def xys2geos(self, xys):
        '''
        多个图像坐标转地理坐标
        '''
        transform = self.dataset.GetGeoTransform()
        geos = np.zeros(xys.shape)
        geos[:, 0] = transform[0] + xys[:, 0] * transform[1] + xys[:, 1] * transform[2]
        geos[:, 1] = transform[3] + xys[:, 0] * transform[4] + xys[:, 1] * transform[5]
        return geos   # (n,2)

    def xy2geo(self, xy):
        '''
        一个图像坐标转地理坐标
        '''
        transform = self.dataset.GetGeoTransform()
        geo_x = transform[0] + xy[0] * transform[1] + xy[1] * transform[2]
        geo_y = transform[3] + xy[0] * transform[4] + xy[1] * transform[5]
        return (geo_x, geo_y)

    def geo2xy(self, geo):
        '''
        根据地理坐标(经纬度)转为影像图上坐标（行列号）
        :param dataset: GDAL地理数据
        :param geo: 经纬度坐标
        :return: 地理坐标(lon,lat)对应的影像图上行列号(row, col)
        '''
        transform = self.dataset.GetGeoTransform()
        x_origin = transform[0]
        y_origin = transform[3]
        pixel_width = transform[1]
        pixel_height = transform[5]
        x_pix = (geo[0] - x_origin) / pixel_width
        y_pix = (geo[1] - y_origin) / pixel_height
        return (x_pix, y_pix)

    def coordsys(self, dst_dataset, ori_lonlat):
        """
        ori_dataset is self.dataset
        """
        ori_proj = self.dataset.GetProjection()
        dst_proj = dst_dataset.GetProjection()
        transformer = Transformer.from_crs(ori_proj, dst_proj, always_xy=True)
        dst_lonlat = transformer.transform(ori_lonlat[0], ori_lonlat[1])
        return dst_lonlat


class TypeTransform():
    def __init__(self) -> None:
        pass

    def uint16Touint8(self, img):
        """
        img: (band, h, w)
        """
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