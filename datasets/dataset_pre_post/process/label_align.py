import numpy as np
import os.path as osp
from pyproj import Transformer
import math
import cv2
from osgeo import gdal, osr
from .utils import get_imlist, TifInfo


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