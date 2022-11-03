import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
from osgeo import gdal

def is_dir(dir):
    if not os.path.exists(dir): os.makedirs(dir)


class HierarchicalUnion():
    def __init__(self) -> None:
        pass

    def imfilter(self, file):
        return True if file[-4:] in ['.tif', '.img'] else False

    def get_imlist(self, imdir):
        imlist_all = os.listdir(imdir)
        imlist = list(filter(self.imfilter, imlist_all))
        return imlist
    
    def savetif(self, imgpath, dst_imgpath, img):
        dataset = gdal.Open(imgpath)
        data_type = gdal.GDT_Byte
        img_height, img_width, img_band = img.shape
        driver = gdal.GetDriverByName("GTiff")
        dst_dataset = driver.Create(dst_imgpath, img_width, img_height, img_band, data_type)
        if dst_dataset is not None:
            dst_dataset.SetGeoTransform(dataset.GetGeoTransform())
            dst_dataset.SetProjection(dataset.GetProjection())

        for i in range(img_band):
            dst_dataset.GetRasterBand(i + 1).WriteArray(img[:,:,i])
        del dst_dataset

    def union(self, basedir, fusedir, targetdir, replace_label):
        """
        fusedir的replace_label替换0
        """
        img_list = os.listdir(basedir)
        for name in tqdm(img_list, total=len(img_list)):
            basepath = osp.join(basedir, name)
            # from IPython import embed;embed()
            fusedpath = osp.join(fusedir, name)
            targetpath = osp.join(targetdir, name)
            baseimg = cv2.imread(basepath, 0)
            fusedimg = cv2.imread(fusedpath, 0)
            for r in replace_label:
                msk1 = baseimg==0
                msk2 = fusedimg==r
                msk = msk1 * msk2
                baseimg[msk] = fusedimg[msk]
            baseimg = baseimg[:,:,np.newaxis]
            self.savetif(basepath, targetpath, baseimg)

    def pad(self, img, padimg):
        img[img==0] = padimg[img==0]
        return img

    def pad_img(self, orimgdir, padimgdir, dstimgdir):
        """
        分层分类方式合并的图像有黑点，多分类图像的全要素补黑点
        """
        orimglist = self.get_imlist(orimgdir)
        for name in orimglist:
            orimgpath = osp.join(orimgdir, name)
            padimgpath = osp.join(padimgdir, name)
            dstimgpath = osp.join(dstimgdir, name)
            orimg = cv2.imread(orimgpath, 0)
            padimg = cv2.imread(padimgpath, 0)
            padedimg = self.pad(orimg, padimg)
            self.save(orimgpath, dstimgpath, padedimg)


if __name__ == '__main__':
    waterdir = r'/data/data/landset30/origin/index/water/newAWEIsh/5600_open'
    buildingdir = r'/data/data/landset30/newmodels_building/UnetPlusPlus/effb3_dicebce/mergepred'
    wbdir = r'/data/data/landset30/union/new_waterAWEIsh_building'
    multiclassdir = r'/data/data/landset30/multiclass/test_segformer_1107_uint8'
    uniondir = r'/data/data/landset30/newunion/new_wAWEIshb_segformer'
    is_dir(wbdir)
    is_dir(uniondir)
    hunion = HierarchicalUnion()
    hunion.union(waterdir, buildingdir, wbdir, [80])
    hunion.union(wbdir, multiclassdir, uniondir, [10, 20])

    # pad 0
    orimgdir = r'/data/data/landset30/choose_data/clip_pred/new_bwAWEIsh_segformer'
    padimgdir = r'/data/data/landset30/choose_data/clip_pred/multiclass'
    dstimgdir = r'/data/data/landset30/newunion/pad_new_bwAWEIsh_segformer'
    is_dir(dstimgdir)
    hunion.pad_img(orimgdir, padimgdir, dstimgdir)
