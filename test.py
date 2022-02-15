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
from datasets.dataset.SegDataset import SegDataset
from training.metrics import IoU, MeanIoU
from tqdm import tqdm


def convert_label(label, inverse=False):
    label_mapping = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        10: 8,
        11: 9,
        12: 10,
        13: 11,
        14: 12,
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


def graytorgb(img):
    label_mapping_rgb = {0: (34, 31, 32),
                         1: (204, 102, 92),
                         2: (209, 154, 98),
                         3: (217, 208, 106),
                         4: (182, 218, 106),
                         5: (142, 217, 105),
                         6: (140, 194, 130),
                         7: (111, 175, 98),
                         8: (219, 245, 215),
                         9: (186, 255, 180),
                         10: (55, 126, 34),
                         11: (111, 174, 167),
                         12: (145, 96, 38),
                         13: (103, 153, 214),
                         14: (41, 96, 246),
                         15: (34, 31, 32),
                         }
    img_rgb = np.stack(np.vectorize(label_mapping_rgb.get)(img), axis=2).astype('uint8')
    return img_rgb


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
    func = MeanIoU()
    value = func(pred, gt)
    value = torch.mean(_distributed_value(torch.Tensor([value]).cuda())).data.cpu().numpy()
    return value


def predict(cfg, model, dataloader):
    classes = 13
    model.eval()
    hist = np.zeros((classes, classes))
    values = []
    with tqdm(total=len(dataloader), desc='test', disable=False, ncols=0) as pbar:
        for i, batch in enumerate(dataloader):
            imgs = batch['image'].to(cfg.device)
            names = batch['id']
            pred = model(imgs)
            # gt = batch['mask'].cuda()

            # value = metric_func(pred, gt)
            # values.append(value)
            # if cfg.lrank == 0:
            #     print('mean_iou: ', np.mean(values))

            if pred.shape[1] == 1:
                pred = F.sigmoid(pred)
                pred = cal_binary_pred(pred)
            else:
                pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
            

            # if pred.shape != gt.shape:
            #     pred = pred.squeeze(1)

            # hist += fast_hist(pred, gt, classes)
            # f1, _, _ = cal_score(hist)

            for i, pre in enumerate(pred):
                pre = pre.data.cpu().numpy().astype(np.uint8)
                # pre = cv2.resize(pre, (1024,1024))   # resize to origin size
                savepath = osp.join(cfg.outdir, names[i][:-4]+'.png')
                cv2.imwrite(savepath, convert_label(pre, inverse=True))
                # cv2.imwrite(savepath, graytorgb(convert_label(pre, inverse=True)))

            # release memory of cuda
            pred = None

            if cfg.lrank == 0:
                pbar.update()


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
    test_dataset = SegDataset(**cfg.data.test_dataset.init_params)

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
