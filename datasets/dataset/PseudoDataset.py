import os
import os.path as osp
import cv2
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional
from ...training.pseudo_dataset import transforms
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


def check_dir(dir):
    if not os.path.exists(dir): os.makedirs(dir)


class PseudoDataset(Dataset):

    def __init__(
            self,
            images_dir: str,
            masks_dir: Optional[str] = None,
            pse_images_dir: str = None,
            pse_masks_dir: str = None,
            ids_csv: Optional[list] = None,
            pse_ids_csv: Optional[list] = None,
            transform_name: Optional[str] = None,
    ):
        super().__init__()
        ids = self._get_ids(ids_csv)
        self.names = ids if ids is not None else os.listdir(images_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.classes = 14   # valid classes

        # pseudo images and masks
        check_dir(pse_masks_dir)
        pse_ids = self._get_ids(pse_ids_csv)
        self.pse_names = pse_ids if pse_ids is not None else os.listdir(pse_masks_dir)
        self.pse_images_dir = pse_images_dir
        self.pse_masks_dir = pse_masks_dir

        images_path, masks_path = self._gather_paths(images_dir, masks_dir, self.names)
        pse_images_path, pse_masks_path = self._gather_paths(pse_images_dir, pse_masks_dir, self.pse_names)

        self.images_path = []
        self.masks_path = []
        self.images_path.extend(images_path)
        self.images_path.extend(pse_images_path)
        self.masks_path.extend(masks_path)
        self.masks_path.extend(pse_masks_path)
        # filter delete_.tif
        self.masks_path = self._filter_delete(self.masks_path)

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

    def _filter_delete(self, names):
        return [name for name in names if 'delete_' not in name]

    def _gather_paths(self, images_dir, masks_dir, names):
        images_path = []
        masks_path = []
        for name in names:
            images_path.append(osp.join(images_dir, name))
            masks_path.append(osp.join(masks_dir, name))
        return images_path, masks_path

    def _get_ids(self, ids_csv):
        return pd.read_csv(ids_csv)['name'].tolist() if ids_csv and osp.exists(ids_csv) else None

    def __len__(self):
        return len(self.masks_path)

    def __getitem__(self, i):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        name = osp.basename(self.masks_path[i])
        imname = name[:-4] + '.tif'
        maskname = name

        im_data = read_image(osp.join(osp.dirname(self.images_path[i]), imname))
        if '.npy' in maskname:
            mask_data = np.load(self.masks_path[i]).transpose(1,2,0)   # c,h,w->h,w,c
        else:
            mask_data = self.convert_label(cv2.imread(self.masks_path[i], 0))
            mask_data = np.identity(self.classes)[mask_data]

        # read data sample
        sample = dict(
            id=maskname,
            image=im_data,
            mask=mask_data)
        # apply augmentations
        if self.transform is not None:
            sample = self.transform(**sample)
        sample['mask'] = pytorchtrans.ToTensor()(sample['mask'].astype('float32')).long()  # expand first dim for mask
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
