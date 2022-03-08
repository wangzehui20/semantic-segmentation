import os.path as osp
import os
import numpy as np
import cv2
from scipy import ndimage
from osgeo import gdal
from tqdm import tqdm


def is_dir(dir):
    if not os.path.exists(dir): os.makedirs(dir)


def convert_label(label, inverse=False):
    label_mapping = {
        0: 0,
        255: 1,
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


def save_tif(dataset, img, dstimg_path, clipul_lonlat=None):
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
    driver = gdal.GetDriverByName("GTiff")
    dst_dataset = driver.Create(dstimg_path, img_width, img_height, img_band, data_type)
    if dst_dataset:
        trans = list(dataset.GetGeoTransform())
        if clipul_lonlat:
            trans[0], trans[3] = clipul_lonlat[0], clipul_lonlat[1]
            trans = tuple(trans)
        dst_dataset.SetGeoTransform(trans)
        dst_dataset.SetProjection(dataset.GetProjection())

    for i in range(img_band):
        dst_dataset.GetRasterBand(i + 1).WriteArray(img[i])
    del dst_dataset


def read_image2(filename):
    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    del dataset

    if len(im_data.shape) == 2:
        return im_data


def get_edge_img(oridir, dstdir):
    img_list = os.listdir(oridir)
    for name in tqdm(img_list, total=len(img_list)):
        img_path = osp.join(oridir, name)
        # dataset = gdal.Open(img_path)
        dst_path = osp.join(dstdir, name)
        img = convert_label(cv2.imread(img_path, 0))
        img_dis = ndimage.morphology.distance_transform_edt(img)
        img_dis[img_dis > 2] = 0
        img_dis[img_dis > 0] = 255
        img = img_dis.astype(np.uint8)
        # img = img[np.newaxis,:,:]
        # save_tif(dataset, img, dst_path)
        cv2.imwrite(dst_path, img)


if __name__ == '__main__':
    oridir = r'/data/dataset/change_detection/merge/256_128/2012/mask'
    dstdir = r'/data/dataset/change_detection/merge/256_128/2012/edge'
    is_dir(dstdir)
    get_edge_img(oridir, dstdir)
