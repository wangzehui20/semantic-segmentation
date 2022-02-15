import os
import os.path as osp
import numpy as np
import random
import torch
import torch.nn as nn
import addict
import training
import pandas as pd
from training.config import parse_config, save_config
from core.mmodel.mmodel_getter import SegmentationScale
from torch.nn import DataParallel
from training.runner import Runner


def worker_init_fn(seed):
    seed = (seed + 1)
    np.random.seed(seed)
    random.seed(seed)
    random.Random().seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_seed(seed=0):
    # forbidden hash random
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    random.Random().seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # pre-optimizer net
    torch.backends.cudnn.benchmark = False
    # conv return is same
    torch.backends.cudnn.deterministic = True


# is it has valid path
def kep_path(keppath):
    path = osp.dirname(keppath)
    kepname = osp.basename(keppath)
    file_list = os.listdir(path)

    for file_name in file_list:
        if kepname in file_name:
            return osp.join(path, file_name)
    return ""


def main(cfg):
    # --------------------------------------------------
    # set GPUs
    # --------------------------------------------------

    if cfg.distributed:
        torch.distributed.init_process_group(backend='nccl')
        cfg.lrank = torch.distributed.get_rank()

        print(f"--------------{cfg.lrank}")
        torch.cuda.set_device(cfg.lrank)
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, cfg.gpus)) if cfg.get('gpus') else ""

    # save config
    logdir = cfg.get('logdir', None)
    if cfg.lrank == 0 and logdir is not None:
        save_config(cfg.to_dict(), logdir, name='config.yaml')
        print(f"Config saved to: {logdir}")

    # --------------------------------------------------
    # define model
    # --------------------------------------------------

    print('Creating model...')
    print(cfg.model.architecture)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import getters
    model = getters.get_model(architecture=cfg.model.architecture, init_params=cfg.model.init_params)

    cfg.model.preweightpath = kep_path(cfg.model.preweightpath)
    if osp.isfile(cfg.model.preweightpath):
        print("Model load from -> ", cfg.model.preweightpath)

        state_dict = torch.load(cfg.model.preweightpath, map_location=torch.device("cpu"))["state_dict"]
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        # for k, v in state_dict.items():
        #     if 'backbones' in k:
        #         k = k.replace('backbones', 'model.backbones')
        #         k = k.replace('proj', 'projection')

    
    if cfg.model.model_scale != 1:
        print("Model scele is ", cfg.model.model_scale)
        model = SegmentationScale(model, scale=cfg.model.model_scale)

    print("Moving model to device...")
    model.to(device)

    if cfg.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    print('Collecting model parameters...') 
    params = model.parameters()

    # --------------------------------------------------
    # define datasets and dataloaders
    # --------------------------------------------------

    print('Creating datasets and loaders..')

    train_dataset = getters.get_dataset(
        name=cfg.data.train_dataset.name,
        init_params=cfg.data.train_dataset.init_params,
    )

    train_sampler = None
    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, **cfg.data.train_dataloader, sampler=train_sampler, shuffle=train_sampler is None,
        worker_init_fn=worker_init_fn
    )

    valid_dataset = getters.get_dataset(
        name=cfg.data.valid_dataset.name,
        init_params=cfg.data.valid_dataset.init_params,
    )

    valid_sampler = None
    if cfg.distributed:
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, **cfg.data.valid_dataloader, sampler=valid_sampler,
    )

    unlabeled_dataset = getters.get_dataset(
        name=cfg.data.unlabeled_dataset.name,
        init_params=cfg.data.unlabeled_dataset.init_params,
    )

    unlabeled_sampler = None
    if cfg.distributed:
        unlabeled_sampler = torch.utils.data.distributed.DistributedSampler(unlabeled_dataset)

    unlabeled_dataloader = torch.utils.data.DataLoader(
        unlabeled_dataset, **cfg.data.unlabeled_dataloader, sampler=unlabeled_sampler,
    )


    # --------------------------------------------------
    # define losses and metrics functions
    # --------------------------------------------------

    losses = {}
    for output_name in cfg.training.losses.keys():
        loss_name = cfg.training.losses[output_name].name
        loss_init_params = cfg.training.losses[output_name].init_params
        losses[output_name] = getters.get_loss(loss_name, loss_init_params).to(device)

    
    metrics = {}
    for output_name in cfg.training.metrics.keys():
        metrics[output_name] = []
        for metric in cfg.training.metrics[output_name]:
            metrics[output_name].append(
                getters.get_metric(metric.name, metric.init_params)
            )

    # --------------------------------------------------
    # define optimizer and scheduler
    # --------------------------------------------------

    print('Defining optimizers and schedulers..')

    optimizer = getters.get_optimizer(
        cfg.training.optimizer.name,
        model_params=params,
        init_params=cfg.training.optimizer.init_params
    )

    if cfg.training.get('scheduler', None):
        scheduler = getters.get_scheduler(
            cfg.training.scheduler.name,
            optimizer,
            cfg.training.scheduler.init_params,
        )
    else:
        scheduler = None

    # --------------------------------------------------
    # define callbacks
    # --------------------------------------------------

    print('Defining callbacks..')
    callbacks = []
    # add scheduler callback
    if scheduler is not None:
        if cfg.training.scheduler.monitor is not None:
            callbacks.append(training.callbacks.Scheduler(
                scheduler,
                sc_name=cfg.training.scheduler.name,
                monitor=cfg.training.scheduler.monitor
            ))
    # add default logging and checkpoint callbacks
    if cfg.logdir is not None:
        # tb logging
        callbacks.append(training.callbacks.TensorBoard(
            osp.join(cfg.logdir, 'tb')
        ))

        # checkpoint
        callbacks.append(training.callbacks.ModelCheckpoint(
            directory=osp.join(cfg.logdir, 'checkpoints'),
            monitor ="val_mask_" + metrics['mask'][0].__name__,
            save_best=True,
            save_top_k=cfg.logging.save_top,
            mode='max',
            verbose=True
        ))

    # --------------------------------------------------
    # model parallel and mix train
    # --------------------------------------------------

    if cfg.distributed:
        print("Creating distributed Model on gpus:", cfg.lrank)
        model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[cfg.lrank])
    else:
        print("Creating DataParallel Model on gpus:", cfg.gpus)
        model = DataParallel(model).to(device)

    # --------------------------------------------------
    # start training
    # --------------------------------------------------
    print('Start training...')

    runner = Runner(
        model,
        model_device=device,
        local_rank=cfg.lrank,
        fp16=cfg.fp16,
        train_sampler=train_sampler,
        distributed=cfg.distributed,
        **cfg.training.runner,
        unlabeled_dataloader=unlabeled_dataloader,
        pseudo_dataset=cfg.data.pseudo_dataset.init_params,
        pseudo_dataloader=cfg.data.pseudo_dataloader,
    )

    runner.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics
    )

    runner.fit(
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        callbacks=callbacks,
        logdir=cfg.logdir,
        **cfg.training.fit
    )


if __name__ == '__main__':
    set_seed(0)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default='config/unet_res50.yaml',
                        type=str, dest='configs', help='The file of the hyper parameters')
    parser.add_argument('--pretrain', default=None,
                        type=str, dest='pretrain', help='pretrain model')
    parser.add_argument('--local_rank', default=-1,
                        type=int, dest='local_rank', help='local rank of current process')
    args = parser.parse_args()
    
    print("Config -> ", args.configs)
    cfg = addict.Dict(parse_config(config=args.configs))
            
    if args.pretrain is not None and osp.isfile(args.pretrain):
        cfg.model.preweight = args.pretrain

    main(cfg)
    os._exit(0)


    