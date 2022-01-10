import argparse
import addict
import torch
import os
import os.path as osp
import torch.nn as nn
import cv2
import numpy as np
import getters
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn import DataParallel
from testing import ttach
from testing.ttach.wrappers import SegmentationTTAWrapper
from training.config import parse_config
from core.mmodel.mmodel_getter import SegmentationScale
from datasets.dataset.SegDataset import SegDataset, SegDataset_6
from datasets.dataset.ChangeDataset import ChangeDataset_label1
from training.metrics import IoU


def convert_label(label, inverse=False):
    label_mapping = {
        0: 0,
        255: 1
    }

    tmp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[tmp == k] = v
    else:
        for k, v in label_mapping.items():
            label[tmp == k] = v
        label[label > len(label_mapping) - 1] = 0
    return label


def _distributed_value(tensor):
    output_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensor, tensor)
    concat = torch.cat(output_tensor, dim=0)
    return concat


def cal_binary_pred(pred):
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    return pred


def metric_func(pred, gt):
    func = IoU()
    value = func(pred, gt)
    value = torch.mean(_distributed_value(torch.Tensor([value]).cuda())).data.cpu().numpy()
    return value


def predict(cfg, model, dataloader):
    classes = 2
    model.eval()
    hist = np.zeros((classes, classes))
    values = []
    for i, batch in enumerate(dataloader):
        imgs = batch['image'].to(cfg.device)
        names = batch['id']
        pred = model(imgs)
        if pred.shape[1] == 1:
            # pred = F.sigmoid(pred)
            pred = cal_binary_pred(pred)
        else:
            pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
        gt = batch['mask'].cuda()
        # if pred.shape != gt.shape:
        #     pred = pred.squeeze(1)

        # hist += fast_hist(pred, gt, classes)
        # f1, _, _ = cal_score(hist)

        value = metric_func(pred, gt)
        values.append(value)
        if cfg.lrank == 0:
            print(i + 1, '/', len(dataloader), ' iou: ', np.mean(values))

        for i, pre in enumerate(pred):
            pre = pre.data.cpu().numpy().squeeze(0).astype(np.uint8)
            # pre = cv2.resize(pre, (1024,1024))   # resize to origin size
            savepath = osp.join(cfg.outdir, names[i][:-4]+'.png')
            cv2.imwrite(savepath, convert_label(pre, inverse=True))

        # release memory of cuda
        pred = None


def model_from_config(cfg):
    model = getters.get_model(architecture=cfg.model.architecture, init_params=cfg.model.init_params)
    if cfg.model.model_scale != 1:
        print("Model scele is ", cfg.model.model_scale)
        model = SegmentationScale(model, scale=cfg.model.model_scale)

    state_dict = torch.load(cfg.model.model_path)['state_dict']
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    return model


def main(cfg):
    # --------------------------------------------------
    # set GPUs
    # --------------------------------------------------

    if cfg.distributed:
        torch.distributed.init_process_group('nccl')
        cfg.lrank = torch.distributed.get_rank()

        print(f"--------------{cfg.lrank}")
        torch.cuda.set_device(cfg.lrank)
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVIDE'] = ",".join(map(str, cfg.gpus)) if cfg.get('gpus') else ""

    # --------------------------------------------------
    # load model
    # --------------------------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.device = device
    model = model_from_config(cfg)

    # tta
    if cfg.tta:
        print("Model in TTA")
        model = SegmentationTTAWrapper(model, ttach.aliases.multiscale_transform([0.5]), merge_mode='mean')
        # model = SegmentationTTAWrapper(model, ttach.aliases.d21_transform(), merge_mode='mean')

    print("Moving model to device...")
    model.to(device)

    # --------------------------------------------------
    # load dataset
    # --------------------------------------------------

    print('Creating datasets and loaders..')
    test_dataset = SegDataset_6(images_dir=cfg.data.test_dataset.init_params.images_dir,
                            masks_dir=cfg.data.test_dataset.init_params.masks_dir)

    test_sampler = None
    if cfg.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, **cfg.data.test_dataloader, sampler=test_sampler
    )

    # --------------------------------------------------
    # model parallel and mix test
    # --------------------------------------------------

    if cfg.distributed:
        print("Creating distributed Model on gpus:", cfg.lrank)
        model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[cfg.lrank])
    else:
        print("Creating DataParallel Model on gpus:", cfg.gpus)
        model = DataParallel(model).to(device)

    # --------------------------------------------------
    # start testing
    # --------------------------------------------------

    predict(cfg, model, test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default='config/unet_res50.yaml',
                        type=str, dest='configs', help="configure file")
    parser.add_argument('--local_rank', default=-1,
                        type=int, dest='local_rank', help="local rank")
    parser.add_argument('--model_path', default='',
                        type=str, dest='model_path', help="model to evaluate test dataset")
    parser.add_argument('--out', default='',
                        type=str, dest='out', help='model predict results')
    parser.add_argument('--mode', default='test',
                        type=str, dest='mode', choices=['test', 'val'], help="predict mode")
    args = parser.parse_args()

    print("Config -> ", args.configs)
    cfg = addict.Dict(parse_config(config=args.configs))

    if not osp.exists(args.out):
        os.makedirs(args.out)
    cfg.outdir = args.out

    if args.model_path is not None:
        cfg.model.model_path = args.model_path
    else:
        print("Model path is invalid")

    main(cfg)

    # val mode
    # if args.mode == 'val':
    # clu_diriou(cfg.out, cfg.data.test_dataset.init_params.images_dir)
