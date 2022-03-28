import os
from osgeo import gdal


def imfilter(file):
    return True if file[-4:] in ['.tif', '.img'] else False


def get_imlist(imdir):
    imlist_all = os.listdir(imdir)
    imlist = list(filter(imfilter, imlist_all))
    return imlist


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


class TifInfo():
    def __init__(self, path):
        self.dt = gdal.Open(path)
        self.w = self.dt.RasterXSize
        self.h = self.dt.RasterYSize
        self.b = self.dt.RasterCount
        self.t = gdal.GetDataTypeName(self.dt.GetRasterBand(1).DataType)
        self.tf = self.dt.GetGeoTransform()
        self.pj = self.dt.GetProjection()
