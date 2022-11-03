import os.path as osp
import os
from osgeo import gdal
from tqdm import tqdm
import numpy as np


def is_dir(dir):
    if not os.path.exists(dir): os.makedirs(dir)

def imfilter(file):
    return True if file[-4:] in ['.tif', '.img'] else False

def get_imlist(imdir):
    imlist_all = os.listdir(imdir)
    imlist = list(filter(imfilter, imlist_all))
    return imlist

def read_image(filename):
    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    del dataset

    if len(im_data) == 3:
        return im_data.transpose([1,2,0])
    else:
        return im_proj, im_geotrans, im_data.transpose([1,2,0])


# ---------------------------------------------
# surface 
# ---------------------------------------------

class SurfaceIndex():
    def __init__(self) -> None:
        pass
        
    def ndbi(self, img):
        '''
        input: float
        return: NDBI = (SWIR – NIR) / (SWIR + NIR)
        '''
        img = img.astype(np.float)
        img_ndbi = (img[:,:,4]-img[:,:,3]) / (img[:,:,4]+img[:,:,3])
        img_ndbi[np.isnan(img_ndbi)] = 0
        return img_ndbi

    def ndvi(self, img):
        '''
        input: float
        return: NDVI = (NIR - R) / (NIR + R)
        '''
        img = img.astype(np.float)
        img_ndvi = (img[:,:,3]-img[:,:,2]) / (img[:,:,3]+img[:,:,2])
        img_ndvi[np.isnan(img_ndvi)] = 0
        return img_ndvi

    def ndi(self, ndi1, ndi2):
        '''
        input: float
        return: NDI = (NDI1 - NDI2) / (NDI1 + NDI2)
        '''
        img_ndi = (ndi1-ndi2) / (ndi1+ndi2)
        img_ndi[np.isnan(img_ndi)] = 0
        return img_ndi

    def pii(self, img):
        '''
        input: float
        return: PII = m*BLUE + n*NIR + C
        '''
        img = img.astype(np.float)
        m = 0.905
        n = 0.435
        c = 0.02
        img_pii = m*img[:,:,0] + n*img[:,:,3] + c
        return img_pii

    def norm(self, img):
        """
        :param img: (h,w,c)
        :return:
        """
        (h, w, c) = img.shape
        img_norm = np.zeros((h,w,c))
        for i in range(c):
            minm = min(img[:,:,i].flatten())
            maxm = max(img[:,:,i].flatten())
            img_norm[:,:,i] = (img[:,:,i]-minm) / (maxm-minm)
        img_norm[np.isnan(img_norm)] = 0
        return img_norm


    def add_band(self, img, imgb):
        (h, w, c) = img.shape
        newimg = np.zeros((h, w, c+1))
        newimg[:,:,:c] = img[:,:,:c]
        newimg[:,:,c] = imgb
        return newimg


def savetif(imgpath, dst_imgpath, img):
    dataset = gdal.Open(imgpath)
    data_type = gdal.GDT_Float32
    img_height, img_width, img_band = img.shape
    driver = gdal.GetDriverByName("GTiff")
    dst_dataset = driver.Create(dst_imgpath, img_width, img_height, img_band, data_type)
    if dst_dataset is not None:
        dst_dataset.SetGeoTransform(dataset.GetGeoTransform())
        dst_dataset.SetProjection(dataset.GetProjection())

    for i in range(img_band):
        dst_dataset.GetRasterBand(i + 1).WriteArray(img[:,:,i])
    del dst_dataset


if __name__ == '__main__':
    imgdir  = r'/data/data/landset30/origin/test_image'
    ndbi_outdir = r'/data/data/landset30/origin/ndbi'
    is_dir(ndbi_outdir)
    surfaceidx = SurfaceIndex()
    imglist = get_imlist(imgdir)
    for img in tqdm(imglist, total=len(imglist)):
        imgpath = osp.join(imgdir, img)
        ndbi_outpath = osp.join(ndbi_outdir, img)
        im_proj, im_geotrans, im_data = read_image(osp.join(imgpath))
        img_ndbi = surfaceidx.norm(surfaceidx.ndbi(im_data))
        img_ndbi = img_ndbi[:,:,np.newaxis]
        savetif(imgpath, ndbi_outpath, img_ndbi)


        

        


        

