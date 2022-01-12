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


class SegDataset(Dataset):

    def __init__(
            self,
            images_dir: str,
            masks_dir: Optional[str] = None,
            ids_csv: Optional[list] = None,
            transform_name: Optional[str] = None,
    ):
        super().__init__()
        ids = self._get_ids(ids_csv)
        self.names = ids if ids is not None else os.listdir(images_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transforms.__dict__[transform_name] if transform_name else None
        self.tfms = pytorchtrans.Compose([pytorchtrans.ToTensor()])
        ignore_label = -1
        self.label_mapping = {
            0: 0,
            80: 1
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

        im_proj, im_geotrans, im_data = read_image(osp.join(self.images_dir, nameid))
        # im_data = read_image(osp.join(self.images_dir, nameid))

        im_data = im_data[:,:,:3]   # rgb
        # im_data = im_data.take([1,2,3],2)
        # im_data = add_band(im_data, norm(ndbi(im_data)))   # building

        # read data sample
        sample = dict(
            id=maskname,
            image=im_data,
            mask=self.convert_label(cv2.imread(osp.join(self.masks_dir, maskname), 0)))
        # apply augmentations
        if self.transform is not None:
            sample = self.transform(**sample)
        sample['mask'] = pytorchtrans.ToTensor()(sample['mask'].astype('float32')).float()  # expand first dim for mask
        # sample['mask'] = sample['mask'][0]
        sample['mask'] = sample['mask']
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


class SegDataset_6(Dataset):
    """
    band number is 6
    """

    def __init__(
            self,
            images_dir: str,
            masks_dir: Optional[str] = None,
            ids_csv: Optional[list] = None,
            transform_name: Optional[str] = None,
    ):
        super().__init__()
        ids = self._get_ids(ids_csv)
        self.names = ids if ids is not None else os.listdir(images_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transforms.__dict__[transform_name] if transform_name else None
        self.tfms = pytorchtrans.Compose([pytorchtrans.ToTensor()])
        ignore_label = -1
        self.label_mapping = {
            0: 0,
            80: 1,
            # 2: 2
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

        im_proj, im_geotrans, im_data = read_image(osp.join(self.images_dir, nameid))

        # read data sample
        sample = dict(
            id=maskname,
            image=im_data,
            mask=self.convert_label(cv2.imread(osp.join(self.masks_dir, maskname), 0)))
        # apply augmentations
        if self.transform is not None:
            sample = self.transform(**sample)
        sample['mask'] = pytorchtrans.ToTensor()(sample['mask'].astype('float32')).float()  # expand first dim for mask
        # sample['mask'] = sample['mask'][0]
        sample['mask'] = sample['mask']
        sample['image'] = self.tfms(np.ascontiguousarray(sample['image']).astype('float32')).float()
        sample['image'] -= torch.tensor([128.0, 128.0, 128.0, 128.0, 128.0, 128.0]).reshape(6, 1, 1)
        sample['image'] /= torch.tensor([128.0, 128.0, 128.0, 128.0, 128.0, 128.0]).reshape(6, 1, 1)

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
