import segmentation_models_pytorch as smp
import torch
import torch.optim
from datasets import dataset
from core.mmseg.mmseg_getter import get_mmseg_model, mmseg_contain
from core.mmodel.mmodel_getter import get_mymodel, mymodel_contain
from training import losses, metrics, optimizers, schedulers
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def get_model(architecture, init_params):
    init_params = init_params or {}
    if mymodel_contain(architecture):
        return get_mymodel(architecture, **init_params)
    # is it in mmseg
    elif mmseg_contain(architecture):
        return get_mmseg_model(architecture, **init_params)
    else:
        print(architecture)
        model_class = smp.__dict__[architecture]
        return model_class(**init_params)


def get_dataset(name, init_params, **kwargs):
    init_params = init_params or {}
    dataset_class = dataset.__dict__[name]
    return dataset_class(**init_params, **kwargs)


def get_loss(name, init_params):
    init_params = init_params or {}
    loss_class = losses.__dict__[name]
    return loss_class(**init_params)


def get_metric(name, init_params):
    init_params = init_params or {}
    metric_class = metrics.__dict__[name]
    return metric_class(**init_params)


def get_optimizer(name, model_params, init_params):
    init_params = init_params or {}
    if name == 'SGD':
        return torch.optim.SGD(model_params, **init_params)
    elif name == 'adamw':
        return torch.optim.AdamW(model_params, **init_params)
    optim_class = optimizers.__dict__[name]
    return optim_class(model_params, **init_params)


def get_scheduler(name, optimizer, init_params):
    init_params = init_params or {}
    print("initial lr: ", optimizer.defaults['lr'])
    if name == 'cosineAnnWarm':
        scheduler = CosineAnnealingWarmRestarts(optimizer, **init_params)
    else:
        scheduler_class = schedulers.__dict__[name]
        scheduler = scheduler_class(optimizer, **init_params)
    return scheduler
