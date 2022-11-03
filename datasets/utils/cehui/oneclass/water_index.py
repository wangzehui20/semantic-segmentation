import numpy as np
import os
import os.path as osp
import cv2
from tqdm import tqdm
from osgeo import gdal
from multiprocessing import Pool


def imfilter(file):
    return True if file[-4:] in ['.tif', '.img'] else False

def get_imlist(imdir):
    imlist_all = os.listdir(imdir)
    imlist = list(filter(imfilter, imlist_all))
    return imlist

def is_dir(dir):
    if not os.path.exists(dir): os.makedirs(dir)


class WaterIndex:
    def __init__(self) -> None:
        pass

    def ndwi(self, img):
        '''
        return: NDWI = (GREEN – NIR) / (GREEN + NIR)
        '''
        img = img.astype(np.float)
        img_ndwi = (img[:,:,1]-img[:,:,3]) / (img[:,:,1]+img[:,:,3])
        img_ndwi[np.isnan(img_ndwi)] = 0
        img_ndwi[np.isinf(img_ndwi)] = 0
        return img_ndwi

    def mndwi(self, img):
        '''
        return: MNDWI = (GREEN – SWIR1) / (GREEN + SWIR1)
        '''
        img = img.astype(np.float)
        img_mndwi = (img[:,:,1]-img[:,:,4]) / (img[:,:,1]+img[:,:,4])
        img_mndwi[np.isnan(img_mndwi)] = 0
        img_mndwi[np.isinf(img_mndwi)] = 0
        return img_mndwi

    def AWEInsh(self, img):
        '''
        return: AWEInsh = 4*(GREEN – SWIR1) / (0.25*NIR + 2.75*SWIR2)
        '''
        img = img.astype(np.float)
        img_nsh = 4*(img[:,:,1]-img[:,:,4]) / (0.25*img[:,:,3]+2.75*img[:,:,5])
        img_nsh[np.isnan(img_nsh)] = 0
        img_nsh[np.isinf(img_nsh)] = 0
        return img_nsh

    def AWEIsh(self, img):
        '''
        return: AWEIsh = BLUE + 2.5*GREEN - 1.5*(NIR+SWIR1) - 0.25*(SWIR2)
        '''
        img = img.astype(np.float)
        img_sh = img[:,:,0] + 2.5*img[:,:,1] - 1.5*(img[:,:,3]+img[:,:,4]) - 0.25*img[:,:,5]
        img_sh[np.isnan(img_sh)] = 0
        img_sh[np.isinf(img_sh)] = 0
        return img_sh


class Method(object):
    def __init__(self, T1=0.05):
        self.T1 = T1

    def is_True(self, a, b):
        """"
        return: a > b -> bool: True
        """
        return a > b

    def compare_vis(self, vis, c):
        """
        :param vis: rgb band
        :param c: c is compare band
        :return: vis > c
        """
        b, g, r = vis[0], vis[1], vis[2]
        msk = self.is_True(b, c) * self.is_True(g, c) * self.is_True(r, c)
        return msk

    def equation_ineq(self, img):
        vis = img[:3]
        nir, swir1, swir2 = img[3], img[4], img[5]
        msk = self.compare_vis(vis, nir) * self.compare_vis(vis, swir1) * self.compare_vis(vis, swir2)
        return msk

    def equation_mag(self, img):
        swir1, swir2 = img[4], img[5]
        return self.is_True(self.T1, swir1) * self.is_True(self.T1, swir2)

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

    def compute(self, img):
        """
        :param img: img has 6 bands -> B G R NIR SWIR1 SWIR2: (c,h,w)
        :return: method1 to extract water
        """
        # img = self.norm(img.transpose(1,2,0)).transpose(2,0,1)
        # return (self.equation_ineq(img) * self.equation_mag(img)).astype(np.uint8)
        return self.equation_ineq(img).astype(np.uint8)


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


def read_image2(filename):
    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    del dataset

    if len(im_data.shape) == 2:
        return im_data


class Evaluate:
    """
    主要用来计算类别f1, precision, recall
    """
    def __init__(self) -> None:
        pass

    def fast_hist(self, pred, gt, nclass):
        mask = (gt >= 0) & (gt < nclass)
        hist = np.bincount(
            nclass * gt[mask].astype(int) +
            pred[mask], minlength=nclass ** 2).reshape(nclass, nclass)
        return hist

    def fscore(self, prec, rec, f):
        '''
        f: 
            f0.5_score: f == 0.5
            f1_score: f == 1
            f2_score: f == 2
        '''
        return (((1+f**2) * prec) * rec) / (f**2 * prec + rec)

    def macroscore(self, hist):
        prec, rec, f1, f0_5 = [], [], [], []
        for i in range(len(hist)):
            prec.append(hist[i,i]/np.sum(hist[:,i]))
            rec.append(hist[i,i]/np.sum(hist[i,:]))
            f1.append(self.fscore(prec[i], rec[i], 1))
            f0_5.append(self.fscore(prec[i], rec[i], 0.5))
        return f1, f0_5, prec, rec

    def microscore(self, hist):
        prec = rec = np.sum(np.diag(hist)) / np.sum(hist)
        f1 = self.fscore(prec, rec, 1)
        f0_5 = self.fscore(prec, rec, 0.5)
        return f1, f0_5, prec, rec


def img_score(labelmerge_dir, predmerge_dir, thred, labelidx):
    classes = 2
    labelist = get_imlist(labelmerge_dir)
    e = Evaluate()
    hist = np.zeros((classes, classes))
    for name in tqdm(labelist, total=len(labelist)):
        labelpath = osp.join(labelmerge_dir, name)
        predpath = osp.join(predmerge_dir, img2png(name))
        # label = cv2.imread(labelpath, 0)
        label = read_image2(labelpath)
        pred = cv2.imread(predpath, 0)
        label[label!=labelidx] = 0
        label[label==labelidx] = 1
        pred[pred!=labelidx] = 0
        pred[pred==labelidx] = 1
        hist += e.fast_hist(pred, label, classes)
    f1, f0_5, precision, recall = e.macroscore(hist)
    print(f"thred: {thred}, f1: {f1}, f0_5: {f0_5}, precision: {precision}, recall: {recall}")


def savetif(imgpath, dst_imgpath, img):
    dataset = gdal.Open(imgpath)
    data_type = gdal.GDT_Byte
    # data_type = gdal.GDT_Float32
    img_height, img_width, img_band = img.shape
    driver = gdal.GetDriverByName("GTiff")
    dst_dataset = driver.Create(dst_imgpath, img_width, img_height, img_band, data_type)
    if dst_dataset is not None:
        dst_dataset.SetGeoTransform(dataset.GetGeoTransform())
        dst_dataset.SetProjection(dataset.GetProjection())

    for i in range(img_band):
        dst_dataset.GetRasterBand(i + 1).WriteArray(img[:,:,i])
    del dst_dataset


def img2png(img):
    return img[:-4] + '.png'


# --------------------------------------
# Water Index method
# --------------------------------------

# def img_index(imgdir, labeldir, dstdir, thred, labelidx, operation=False):
#     imglist = get_imlist(imgdir)
#     widx = WaterIndex()
#     for img in imglist:
#         imgpath = osp.join(imgdir, img)
#         dstpath = osp.join(dstdir, img)
#         im_proj, im_geotrans, im_data = read_image(imgpath)
#         img_wateridx = widx.ndwi(im_data)
#         img_wateridx[img_wateridx>=thred] = labelidx
#         img_wateridx[img_wateridx<thred] = 0
#         img_wateridx = img_wateridx.astype(np.uint8)
#         if operation:
#             img_wateridx = open_operation(img_wateridx, labelidx)   # open operation
#         savetif(imgpath, dstpath, img_wateridx[:,:,np.newaxis])
#     img_score(labeldir, dstdir, thred, labelidx)
#
#
# def img_index_multi(imgdir, labeldir, dstdir, threds, labelidx, operation=False):
#     inputs = []
#     for i in range(len(threds)):
#         if operation:
#             dstdir = osp.join(dstdir, str(threds[i]).replace('.', '_')+'_open')   # open operation
#         else:
#             dstdir = osp.join(dstdir, str(threds[i]).replace('.', '_'))
#         inputs.append([imgdir, labeldir, dstdir, threds[i], labelidx, operation])
#
#     pool = Pool()
#     pool.starmap(img_index, inputs)


def open_operation(img, labelidx):
    img[img==labelidx] = 255
    kernel = np.ones((3,3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img[img==255] = labelidx
    return img


# --------------------------------------
# Water Method
# --------------------------------------

def convert_label(label, inverse=False):
    label_mapping = {
        0: 0,
        60: 1
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

def img_index(imgdir, labeldir, dstdir, thred, labelidx, operation=False):
    imglist = get_imlist(imgdir)
    mtd = Method(thred)
    for name in tqdm(imglist, total=len(imglist)):
        imgpath = osp.join(imgdir, name)
        dstpath = osp.join(dstdir, img2png(name))
        im_proj, im_geotrans, im_data = read_image(imgpath)   # im_data: (H,W,C)
        img_mtd = convert_label(mtd.compute(im_data.transpose(2,0,1)), inverse=True)
        if operation:
            img_mtd = open_operation(img_mtd, labelidx)   # open operation
        cv2.imwrite(dstpath, img_mtd)
        # savetif(imgpath, dstpath, img_mtd[:,:,np.newaxis])
    img_score(labeldir, dstdir, thred, labelidx)


def img_index_multi(imgdir, labeldir, dstdir, threds, labelidx, operation=False):
    inputs = []
    for i in range(len(threds)):
        if operation:
            dstdir = osp.join(dstdir, str(threds[i]).replace('.', '_')+'_open')   # open operation
        else:
            dstdir = osp.join(dstdir, str(threds[i]).replace('.', '_'))
        is_dir(dstdir)
        inputs.append([imgdir, labeldir, dstdir, threds[i], labelidx, operation])

    pool = Pool()
    pool.starmap(img_index, inputs)

if __name__ == '__main__':
    labelidx = 60
    operation = False

    imgdir = r'/data/data/landset30/origin/test_image'
    labeldir = r'/data/data/landset30/origin/mergelabel'
    mtdir = r'/data/data/landset30/origin/index/water/method'

    threds = [0.05]
    img_index_multi(imgdir, labeldir, mtdir, threds, labelidx, operation)