import os
import os.path as osp
import torch
import addict
import numpy as np
from tqdm import tqdm
from training.config import parse_config

import getters
from torchvision import transforms as pytorchtrans
from torch.utils.data import Dataset
import pandas as pd
from typing import Optional
# from datasets import transforms
# from datasets import read_image
from datasets.dataset.SegDataset import SegDataset, SegEdgeDataset
import random

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class MultiSeg(Dataset):

#     def __init__(
#         self,
#         images_dir: str,
#         ids: Optional[list] = None,
#         transform_name: Optional[str] = None,
#     ):
#         super().__init__()

#         self.names = ids if ids is not None else os.listdir(images_dir)
#         self.images_dir = images_dir
#         self.transform = transforms.__dict__[transform_name] if transform_name else None
#         self.tfms = pytorchtrans.Compose([pytorchtrans.ToTensor()])
#         ignore_label = -1

#     def __len__(self):
#         return len(self.names)

#     def __getitem__(self, i):
#         name = self.names[i]
#         nameid = name   # .tif
#         maskname = name[:-4] + '.png'

#         # read data sample
#         sample = dict(
#             id=maskname,
#             image=read_image(osp.join(self.images_dir, nameid)),
#             )          
#         # apply augmentations
#         if self.transform is not None:
#             sample = self.transform(**sample)
#         sample['image'] = self.tfms(np.ascontiguousarray(sample['image']).astype('float32')).float()
#         sample['image'] -= torch.tensor([128.0, 128.0, 128.0]).reshape(3,1,1)
#         sample['image'] /= torch.tensor([128.0, 128.0, 128.0]).reshape(3,1,1)
#         return sample['image']


def model_from_config(path: str, checkpoint_path: str):
    """Create model from configuration specified in config file and load checkpoint weights"""
    cfg = addict.Dict(parse_config(config=path))  # read and parse config file
    init_params = cfg.model.init_params  # extract model initialization parameters

    if "encoder_weights" in init_params:
        init_params["encoder_weights"] = None  # because we will load pretrained weights for whole model

    # init_params['ispretrain'] = False
    model = getters.get_model(architecture=cfg.model.architecture, init_params=init_params)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)["state_dict"]
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    return model


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def save_checkpoint(dir, model_dict, **kwargs):
    state = {
        'state_dict': model_dict,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'swa_convert.pth')
    torch.save(state, filepath)


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input in tqdm(loader):
        input = input["image"].cuda()
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def kep_path(keppath):
    path = os.path.dirname(keppath)
    kepname = os.path.basename(keppath)
    file_list = os.listdir(path)

    for file_name in file_list:
        if kepname in file_name:
            return os.path.join(path, file_name)
    return ""


def getcosindex(range_select):
    out = []
    for pthname in range_select:
        out.append(kep_path(pthname))
    return out

def worker_init_fn(seed):
    seed = (seed + 1)
    np.random.seed(seed)
    random.seed(seed)
    random.Random().seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":

    cfg = addict.Dict(parse_config(config=r'/data/code/semantic-segmentation-semi-supervised-learning/config/change_detection/train/split/effb3_dicebce_scse_edge.yaml'))

    dataset = SegEdgeDataset(**cfg.data.train_dataset.init_params)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    configpath = r'/data/code/semantic-segmentation-semi-supervised-learning/config/change_detection/train/split/effb3_dicebce_scse_edge.yaml'
    select_index = [
        '/data/data/change_detection/models/train/split/unet/effb3_dicebce_scse_edge_size160/checkpoints/k-ep[24]-0.9296.pth',
        '/data/data/change_detection/models/train/split/unet/effb3_dicebce_scse_edge_size160/checkpoints/k-ep[45]-0.9295.pth',
        '/data/data/change_detection/models/train/split/unet/effb3_dicebce_scse_edge_size160/checkpoints/k-ep[54]-0.9288.pth',
        '/data/data/change_detection/models/train/split/unet/effb3_dicebce_scse_edge_size160/checkpoints/k-ep[41]-0.9285.pth']
    name_list = getcosindex(select_index)
    print(name_list)

    modelpath = r"/data/data/change_detection/models/train/split/unet/effb3_dicebce_scse_edge_size160/swa/checkpoints"   # save model
    swa_model = model_from_config(configpath, "")
    swa_model = swa_model.to(device)

    # base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    # opt = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)

    from torchcontrib.optim import SWA

    for i, name in enumerate(name_list):
        print(i, '/', len(name_list), "=================")
        print("load -> ", name)
        model = model_from_config(configpath, name)
        model = model.to(device)
        print("union -> ", name)
        moving_average(swa_model, model, 1.0 / (i + 1))

    SWA.bn_update(dataloader, swa_model, device)

    save_checkpoint(modelpath, swa_model.state_dict())
    print("swa finish")







