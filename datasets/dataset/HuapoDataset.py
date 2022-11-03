import os
import os.path as osp
import cv2
import numpy as np
import torch
import pandas as pd
import h5py
import torchvision.transforms
from torch.utils.data import Dataset
from typing import Optional
from . import transforms
from torchvision import transforms as pytorchtrans

import warnings

warnings.simplefilter("ignore")


class HuapoDataset(Dataset):

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
        self.tfms = pytorchtrans.Compose([pytorchtrans.ToTensor(), pytorchtrans.Normalize((0.9227, 0.9541, 0.9596), (0.1307, 0.2204, 0.3932))])
        ignore_label = -1
        self.label_mapping = {
            0: 0,
            255: 1,
        }

    def _get_ids(self, ids_csv):
        return pd.read_csv(ids_csv)['name'].tolist() if ids_csv and osp.exists(ids_csv) else None

    def _get_data(self, datapath, flag=0):
        # flag: 0 is image, 1 is mask
        f = h5py.File(datapath, 'r')
        data = f['img'][:] if not flag else f['mask'][:]
        f.close()
        return data

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        name = self.names[i]
        nameid = name
        maskname = name.replace('image', 'mask')

        # im_data = read_image(osp.join(self.images_dir, nameid))
        im_data = self._get_data(osp.join(self.images_dir, nameid)).transpose(2,0,1)[1:4].transpose(1,2,0)
        if self.masks_dir is not None:
            mask_data = self._get_data(osp.join(self.masks_dir, maskname), 1)
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
                id=maskname,
                image=im_data,)
        sample['image'] = self.tfms(np.ascontiguousarray(sample['image']).astype('float32')).float()

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

# dataset into 5 fold
class HuapoDataset2(Dataset):

    def __init__(
            self,
            images_dir: str,
            ids: Optional[list] = None,
            transform_name: Optional[str] = None,
            mask_onehot: bool = False
    ):
        super().__init__()
        self.names = ids
        self.images_dir = images_dir
        self.mask_onehot = mask_onehot
        self.classes = 1
        self.transform = transforms.__dict__[transform_name] if transform_name else None
        self.tfms = pytorchtrans.Compose([pytorchtrans.ToTensor(), pytorchtrans.Normalize((0.9227, 0.9541, 0.9596), (0.1307, 0.2204, 0.3932))])
        ignore_label = -1
        self.label_mapping = {
            0: 0,
            255: 1,
        }

    def _get_data(self, datapath, flag=0):
        # flag: 0 is image, 1 is mask
        f = h5py.File(datapath, 'r')
        data = f['img'][:] if not flag else f['mask'][:]
        f.close()
        return data

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        name = self.names[i]
        nameid = name
        maskname = name.replace('image', 'mask')

        # im_data = read_image(osp.join(self.images_dir, nameid))
        im_data = self._get_data(osp.join(self.images_dir, 'img', nameid)).transpose(2,0,1)[1:4].transpose(1,2,0)
        if osp.join(self.images_dir, 'mask') is not None:
            mask_data = self._get_data(osp.join(self.images_dir, 'mask', maskname), 1)
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
                id=maskname,
                image=im_data,)
        sample['image'] = self.tfms(np.ascontiguousarray(sample['image']).astype('float32')).float()

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
