import cv2
import numpy as np
import os
import os.path as osp
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from tqdm import tqdm
from osgeo import gdal

def read_image(filename):
    dataset = gdal.Open(filename)

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵

    del dataset
    if len(im_data) == 3:
        return im_data.transpose([1, 2, 0])
    else:
        return im_proj, im_geotrans, im_data.transpose([1, 2, 0])

def mse_psnr_ssim_dir(im_trans_dir, im_ref_dir):
    # ref_names = sorted(os.listdir(im_ref_dir))
    ref_names = set(sorted(os.listdir(im_trans_dir))) # -------------------------------------------
    mses, psnrs, ssims = [], [], []
    for name in tqdm(ref_names, total=len(ref_names)):
        # im_trans_path = osp.join(im_trans_dir, name.replace('2016_', '2012_'))
        im_trans_path = osp.join(im_trans_dir, name)
        im_ref_path = osp.join(im_ref_dir, name)
        # im_trans = cv2.imread(im_trans_path)
        # im_ref = cv2.imread(im_ref_path)
        im_trans = read_image(im_trans_path)
        im_ref = read_image(im_ref_path)
        # nodata
        im_ref[im_ref==256] = 0
        # from IPython import embed; embed()
        MSE, PSNR, SSIM = mse_psnr_ssim(im_trans[:-128,:-128,:], im_ref)
        mses.append(MSE)
        psnrs.append(PSNR)
        ssims.append(SSIM)
    print(np.mean(mses), np.mean(psnrs), np.mean(ssims))

def mse_psnr_ssim(im_trans, im_ref):
    MSE = mean_squared_error(im_trans, im_ref)
    PSNR = peak_signal_noise_ratio(im_trans, im_ref)
    SSIM = structural_similarity(im_trans, im_ref, multichannel=True)
    return MSE, PSNR, SSIM

if __name__ == '__main__':
    im_trans_dir = r'/data/data/change_detection/merge/256_128/2012/trans_bigmap/image_drit_concat_resume'
    im_ref_dir = r'/data/dataset/change_detection/origin_merge/2016/image'
    im_trans_hm_dir = r'/data/data/change_detection/merge/256_128/2012/trans_bigmap/image_hm'
    im_trans_reinhard_dir = r'/data/data/change_detection/merge/256_128/2012/trans_bigmap/image_reinhard'
    # im_trans_ugatit_dir = r'/data/data/update/256_128/train/image_ugatit_9res_5layer'
    im_trans_unit_dir = r'/data/data/change_detection/merge/256_128/2012/trans_bigmap/image_unit'
    im_trans_cyclegan_dir = r'/data/data/change_detection/merge/256_128/2012/trans_bigmap/image_cyclegan'
    # print('image_drit')
    # mse_psnr_ssim_dir(im_trans_dir, im_ref_dir)
    # print('image_hm')
    # mse_psnr_ssim_dir(im_trans_hm_dir, im_ref_dir)
    # print('image_reinhard')
    # mse_psnr_ssim_dir(im_trans_reinhard_dir, im_ref_dir)
    # # print('image_ugatit_9res_5layer')
    # # mse_psnr_ssim_dir(im_trans_ugatit_dir, im_ref_dir)
    # print('image_unit')
    # mse_psnr_ssim_dir(im_trans_unit_dir, im_ref_dir)
    print('image_cyclegan')
    mse_psnr_ssim_dir(im_trans_cyclegan_dir, im_ref_dir)