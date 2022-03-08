import os
import os.path as osp
import cv2
import numpy as np
import torch
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
        return im_data.transpose([1,2,0])
    else:
        return im_proj, im_geotrans, im_data.transpose([1,2,0])


class ChangeDataset(Dataset):

    def __init__(
        self,
        images_dir: str,
        masks_dir: Optional[str] = None,
        edge_dir: Optional[str] = None,
        ids: Optional[list] = None,
        transform_name: Optional[str] = None,
    ):
        super().__init__()

        self.names = ids if ids is not None else os.listdir(images_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.edge_dir = edge_dir
        self.transform = transforms.__dict__[transform_name] if transform_name else None
        self.tfms = pytorchtrans.Compose([pytorchtrans.ToTensor()])
        ignore_label = -1
        self.label_mapping = {
            0: 0,
            255: 1
        }

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        name = self.names[i]
        nameid = name   # .tif
        maskname = name
        edgename = name

        # im_proj, im_geotrans, im_data = read_image(osp.join(self.images_dir, nameid))
        im_data = read_image(osp.join(self.images_dir, nameid))

        # im_data = im_data[:,:,:3]   # rgb
        # im_data = im_data.take([1,2,3],2)
        # im_data = add_band(im_data, norm(ndbi(im_data)))   # building

        # read data sample
        # from IPython import embed;embed()
        sample = dict(
            id=maskname,
            image=im_data,
            mask=self.convert_label(cv2.imread(osp.join(self.masks_dir, maskname), 0)),
            edge=self.convert_label(cv2.imread(osp.join(self.edge_dir, edgename), 0)),
        )
        # apply augmentations
        if self.transform is not None:
            sample = self.transform(**sample)
        sample['mask'] = pytorchtrans.ToTensor()(sample['mask'].astype('float32')).float() # expand first dim for mask
        sample['edge'] = pytorchtrans.ToTensor()(sample['edge'].astype('float32')).float()  # expand first dim for mask
        # sample['mask'] = sample['mask'][0]
        sample['image'] = self.tfms(np.ascontiguousarray(sample['image']).astype('float32')).float()
        sample['image'] -= torch.tensor([128.0, 128.0, 128.0]).reshape(3,1,1)
        sample['image'] /= torch.tensor([128.0, 128.0, 128.0]).reshape(3,1,1)

        return sample

    def convert_label(self, label, inverse=False):
        label_mapping = self.label_mapping
        tmp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[tmp==k] = v
        else:
            for k, v in self.label_mapping.items():
                label[tmp==k] = v
            label[label>len(label_mapping)-1] = 0
        return label


class ChangeDataset_input2(Dataset):

    def __init__(
        self,
        images1_dir: str,
        images2_dir: str,
        masks1_dir: Optional[str] = None,
        masks2_dir: Optional[str] = None,
        edge1_dir: Optional[str] = None,
        edge2_dir: Optional[str] = None,
        ids: Optional[list] = None,
        transform_name: Optional[str] = None,
    ):
        super().__init__()

        self.names = ids if ids is not None else os.listdir(images1_dir)
        self.images1_dir = images1_dir
        self.images2_dir = images2_dir
        self.masks1_dir = masks1_dir
        self.masks2_dir = masks2_dir
        self.edge1_dir = edge1_dir
        self.edge2_dir = edge2_dir

        self.transform = transforms.__dict__[transform_name] if transform_name else None
        self.tfms = pytorchtrans.Compose([pytorchtrans.ToTensor()])
        ignore_label = -1
        self.label_mapping = {
            0: 0,
            255: 1
        }

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        name = self.names[i]
        nameid = name   # .tif
        maskname = name
        edgename = name

        # im_proj, im_geotrans, im_data = read_image(osp.join(self.images_dir, nameid))
        im1_data = read_image(osp.join(self.images1_dir, nameid))
        im2_data = read_image(osp.join(self.images2_dir, nameid))

        # im_data = im_data[:,:,:3]   # rgb
        # im_data = im_data.take([1,2,3],2)
        # im_data = add_band(im_data, norm(ndbi(im_data)))   # building

        # read data sample
        # from IPython import embed;embed()
        sample = dict(
            id=maskname,
            image=im1_data,
            image2=im2_data,
            mask1=self.convert_label(cv2.imread(osp.join(self.masks1_dir, maskname), 0)),
            mask2=self.convert_label(cv2.imread(osp.join(self.masks2_dir, maskname), 0)),
            edge1=self.convert_label(cv2.imread(osp.join(self.edge1_dir, maskname), 0)),
            edge2=self.convert_label(cv2.imread(osp.join(self.edge2_dir, maskname), 0)),
        )

        newsample = dict(
            id=maskname,
        )

        # apply augmentations
        if self.transform is not None:
            sample = self.transform(**sample)
        mask1_data = pytorchtrans.ToTensor()(sample['mask1'].astype('float32')).float() # expand first dim for mask
        mask2_data = pytorchtrans.ToTensor()(sample['mask2'].astype('float32')).float() # expand first dim for mask
        edge1_data = pytorchtrans.ToTensor()(sample['edge1'].astype('float32')).float()  # expand first dim for mask
        edge2_data = pytorchtrans.ToTensor()(sample['edge2'].astype('float32')).float()  # expand first dim for mask
        # sample['mask'] = sample['mask'][0]
        im1_data = self.tfms(np.ascontiguousarray(sample['image']).astype('float32')).float()
        im1_data -= torch.tensor([128.0, 128.0, 128.0]).reshape(3,1,1)
        im1_data /= torch.tensor([128.0, 128.0, 128.0]).reshape(3,1,1)
        im2_data = self.tfms(np.ascontiguousarray(sample['image2']).astype('float32')).float()
        im2_data -= torch.tensor([128.0, 128.0, 128.0]).reshape(3, 1, 1)
        im2_data /= torch.tensor([128.0, 128.0, 128.0]).reshape(3, 1, 1)
        newsample['image'] = [im1_data, im2_data]
        bdata = [mask1_data, edge1_data]
        adata = [mask2_data, edge2_data]
        newsample['mask'] = [bdata, adata]

        return newsample

    def convert_label(self, label, inverse=False):
        label_mapping = self.label_mapping
        tmp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[tmp==k] = v
        else:
            for k, v in self.label_mapping.items():
                label[tmp==k] = v
            label[label>len(label_mapping)-1] = 0
        return label