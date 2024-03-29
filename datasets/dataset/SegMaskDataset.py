import os
import os.path as osp
import cv2
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional
from . import transforms
from torchvision import transforms as pytorchtrans
from osgeo import gdal

import warnings

warnings.simplefilter("ignore")


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


class SegMaskDataset(Dataset):

    def __init__(
            self,
            images_dir: str,
            masks_dir: Optional[str] = None,
            ids_csv: Optional[list] = None,
            transform_name: Optional[str] = None,
            mask_onehot: bool = False
    ):
        super().__init__()
        ids = self._get_ids(ids_csv)
        self.names = ids if ids is not None else os.listdir(images_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.mask_onehot = mask_onehot
        self.classes = 1
        self.transform = transforms.__dict__[transform_name] if transform_name else None
        self.tfms = pytorchtrans.Compose([pytorchtrans.ToTensor()])
        ignore_label = -1
        self.label_mapping = {
            0: 0,
            255: 1,
        }

    def _get_ids(self, ids_csv):
        return pd.read_csv(ids_csv)['name'].tolist() if ids_csv and osp.exists(ids_csv) else None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        name = self.names[i]
        nameid = name  # .tif
        maskname = name

        # im_data = read_image(osp.join(self.images_dir, nameid))
        im_data = cv2.imread(osp.join(self.images_dir, nameid))

        # 颜色迁移图像背景不为0
        # clipimg_dir = r'/data/data/change_detection/merge/256_128/2012/mask'
        # clipimg_data = cv2.imread(osp.join(clipimg_dir, nameid.replace('2016_merge', '2012_merge')))
        # clipimg_dir = r'/data/data/update/256_128/train/mask'
        # clipimg_data = cv2.imread(osp.join(clipimg_dir, nameid.replace('2019_', '2018_')))
        # im_data[clipimg_data>0] = 0

        if self.masks_dir is not None:
            mask_data = self.convert_label(cv2.imread(osp.join(self.masks_dir, maskname), 0))
            if self.mask_onehot:
                mask_data = np.identity(self.classes)[mask_data]
            # read data sample
            sample = dict(
                id=maskname,
                image=im_data,
                mask=mask_data)
            # apply augmentations   
            if self.transform is not None:
                sample = self.transform(**sample)
            sample['mask'] = pytorchtrans.ToTensor()(sample['mask'].astype('float32')).float()  # expand first dim for mask
            # sample['mask'] = sample['mask'][0]
            sample['mask'] = sample['mask']
        else:
            sample = dict(
                id=nameid,
                image=im_data,)
        sample['image'] = self.tfms(np.ascontiguousarray(sample['image']).astype('float32')).float()
        sample['image'] -= torch.tensor([128.0, 128.0, 128.0]).reshape(3, 1, 1)
        sample['image'] /= torch.tensor([128.0, 128.0, 128.0]).reshape(3, 1, 1)
        return sample

    def convert_label(self, label, inverse=False):
        label_mapping = self.label_mapping
        tmp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[tmp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[tmp == k] = v
            label[label > len(label_mapping) - 1] = 0
        return label
