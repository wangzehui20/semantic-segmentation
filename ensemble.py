import os
import os.path as osp
import torch
import addict
import numpy as np
import cv2
import torch.nn.functional as F
from core.mmodel.mmodel_getter import SegmentationScale
from training.config import parse_config
import getters
from torchvision import transforms as pytorchtrans
from torch.utils.data import Dataset
from typing import Optional
from datasets import transforms
from datasets import read_image
from testing import ttach
from testing.ttach.wrappers import SegmentationTTAWrapper
from testing.eval import fast_hist, cal_score
from datasets import LABEL

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiSeg(Dataset):

    def __init__(
        self,
        images_dir: str,
        masks_dir: Optional[str] = None,
        ids: Optional[list] = None,
        transform_name: Optional[str] = None,
    ):
        super().__init__()

        self.names = ids if ids is not None else os.listdir(images_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transforms.__dict__[transform_name] if transform_name else None
        self.tfms = pytorchtrans.Compose([pytorchtrans.ToTensor()])
        ignore_label = -1
        self.label_mapping = {
            0: 0,
            LABEL: 1
        }

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        nameid = name   # .tif
        maskname = name
        # maskname = name.split('.')[0] + '.png'

        im_proj, im_geotrans, im_data = read_image(osp.join(self.images_dir, nameid))
        im_data = im_data[:,:,:3]   # rgb
        # im_data = add_band(im_data, norm(ndwi(im_data)))   # water
        # im_data = add_band(im_data, norm(ndbi(im_data)))   # building
        # im_data = add_band(im_data, norm(ndi(ndbi(im_data), ndvi(im_data))))


        # read data sample
        sample = dict(
            id=maskname,
            image=im_data,
            mask=convert_label(cv2.imread(osp.join(self.masks_dir, maskname),0)))
            
        # apply augmentations
        if self.transform is not None:
            sample = self.transform(**sample)
        sample['mask'] = pytorchtrans.ToTensor()(sample['mask'].astype('float32')).long() # expand first dim for mask
        sample['mask'] = sample['mask'][0]
        sample['image'] = self.tfms(np.ascontiguousarray(sample['image']).astype('float32')).float()
        sample['image'] -= torch.tensor([128.0, 128.0, 128.0]).reshape(3,1,1)
        sample['image'] /= torch.tensor([128.0, 128.0, 128.0]).reshape(3,1,1)
        # sample['image'] -= torch.tensor([128.0, 128.0, 128.0, 128.0, 128.0, 128.0]).reshape(6,1,1)
        # sample['image'] /= torch.tensor([128.0, 128.0, 128.0, 128.0, 128.0, 128.0]).reshape(6,1,1)
        # sample['image'] -= torch.tensor([128.0, 128.0, 128.0, 128.0, 128.0, 128.0, 0.5]).reshape(7,1,1)
        # sample['image'] /= torch.tensor([128.0, 128.0, 128.0, 128.0, 128.0, 128.0, 0.5]).reshape(7,1,1)
        return sample


class EnsembleModel(torch.nn.Module):
    """Ensemble of torch models, pass tensor through all models and average results"""

    def __init__(self, models: list):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        # self.class_weights = class_weights
        # assert len(self.models)==len(self.class_weights) or np.sum(np.array(self.class_weights)) == 1.0

    def forward(self, x):
        preall = None
        for index, model in enumerate(self.models):
            # wt_class = self.class_weights[index]
            pre = model(x)
            # pre = pre #  * wt_class
            if preall is None:
                preall = pre
            else:
                preall += pre
        return preall/len(self.models)


def model_from_config(path: str, checkpoint_path: str):
    """Create model from configuration specified in config file and load checkpoint weights"""
    cfg = addict.Dict(parse_config(config=path))  # read and parse config file
    init_params = cfg.model.init_params  # extract model initialization parameters
    if "encoder_weights" in init_params:
        init_params["encoder_weights"] = None  # because we will load pretrained weights for whole model

    print(init_params)
    # init_params['ispretrain'] = False
    model = getters.get_model(architecture=cfg.model.architecture, init_params=init_params)

    if cfg.model.model_sacle != 1:
        print('moedl sacel is ', cfg.model.model_sacle)
        model = SegmentationScale(model, scale=cfg.model.model_sacle)

    state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})

    return model


def get_model(configpath, modelpath, ids=None, tta=True):
    print("Available devices:", device)

    if isinstance(configpath, list) and isinstance(modelpath, list):
        models = []
        for index, path in enumerate(configpath):
            model = model_from_config(configpath[index], modelpath[index])
            # model = SegmentationTTAWrapper(model, ttach.aliases.d21_transform(), merge_mode='mean')
            models.append(model)
        model = EnsembleModel(models)
    else:
        model = model_from_config(configpath, modelpath)
    
    if tta:
        model = SegmentationTTAWrapper(model, ttach.aliases.d21_transform(), merge_mode='mean')

    model = model.to(device)
    model.eval()
    return model


def predict(model, dataloader, outdir):
    classes = 2
    hist = np.zeros((classes, classes))
    for i, batch in enumerate(dataloader):
        imgs = batch['image'].to(device)
        names = batch['id']
        pred = model(imgs)
        if pred.shape[1] == 1:
            pred = F.sigmoid(pred)
            pred[pred>0.5] = 1
            pred[pred<=0.5] = 0
            pred = pred.data.cpu().numpy().astype(np.uint8)
        else:    
            pred = torch.argmax(F.softmax(pred, dim=1), dim=1).data.cpu().numpy().astype(np.uint8)
        gt = batch['mask'].data.cpu().numpy().astype(np.uint8)
        if pred.shape != gt.shape:
            pred = pred.squeeze(1)
        hist += fast_hist(pred, gt, classes)
        f1, prec, recall = cal_score(hist)
        print(i+1, '/', len(dataloader), ' f1: ', f1)

        for i, pre in enumerate(pred):
            savepath = osp.join(outdir, names[i])
            cv2.imwrite(savepath, convert_label(pre, inverse=True))


def convert_label(label, inverse=False):
    label_mapping = {0: 0,
                    LABEL: 1}

    tmp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[tmp==k] = v
    else:
        for k, v in label_mapping.items():
            label[tmp==k] = v
        label[label>len(label_mapping)-1] = 0
    return label


if __name__ == "__main__":
    tta = True

    images_dir = r'/data/data/landset30/Unet_bifpn/512_128/test/image'
    masks_dir = r'/data/data/landset30/Unet_bifpn/512_128/test/mask'
    test_dataset = MultiSeg(images_dir=images_dir, masks_dir=masks_dir)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    outdir = r'/data/data/landset30'
    configpath = [r'/data/data/multiclass/models/Unet_bifpn/effb3_ce_cosine/config.yaml']
    modelpath = [r"/data/data/multiclass/models/Unet_bifpn/effb3_ce_cosine/swa/checkpoints"]

    model = get_model(configpath, modelpath, tta=tta)
    predict(model, test_dataloader, outdir)


