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


class PseudoDataset(Dataset):

    def __init__(
            self,
            labeled_images_csv: str,
            unlabeled_images_csv: Optional[str] = None,
            transform_name: Optional[str] = None,
            classes = 14
    ):
        super().__init__()
        self.im_path = []
        labeled_impath = self._get_ids(labeled_images_csv)
        unlabeled_impath = self._get_ids(unlabeled_images_csv)
        self.classes = classes
        self.im_path.extend(labeled_impath)
        self.im_path.extend(unlabeled_impath)
        self.transform = transforms.__dict__[transform_name] if transform_name else None
        self.tfms = pytorchtrans.Compose([pytorchtrans.ToTensor()])
        ignore_label = -1
        self.label_mapping = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6,
            7: 7,
            8: 0,
            9: 0,
            10: 8,
            11: 9,
            12: 10,
            13: 11,
            14: 12,
            15: 13,
        }

    def _get_ids(self, ids_csv):
        return pd.read_csv(ids_csv)['name'].tolist() if ids_csv and osp.exists(ids_csv) else None

    def __len__(self):
        return len(self.im_path)

    def __getitem__(self, i):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        name = self.im_path[i].split('/')[-1]
        nameid = name  # .tif
        maskname = name

        im_data = read_image(self.im_path[i])
        mask_path = self.im_path[i].replace('image', 'mask')
        if os.path.exists(mask_path):
            mask_data = self.convert_label(cv2.imread(mask_path, 0))
            mask_data = np.identity(self.classes)[mask_data]
        else:
            mask_data = np.load(mask_path[:-4]+'.npy')

        # read data sample
        sample = dict(
            id=maskname,
            image=im_data,
            mask=mask_data)
        # apply augmentations
        if self.transform is not None:
            sample = self.transform(**sample)
        sample['mask'] = pytorchtrans.ToTensor()(sample['mask'].astype('float32')).long()  # expand first dim for mask
        sample['mask'] = sample['mask'][0]
        # sample['mask'] = sample['mask']
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
