import torch.nn as nn
import torch.nn.functional as F
from core.mmodel.model_zoo.Unet_bifpn import Unet_bifpn
from core.mmodel.model_zoo.SGEPUNet import EPUnet, SGEPUnet
from core.mmodel.model_zoo.UnetEdge import UnetEdge


def get_mymodel(architecture, **init_params):
    print(init_params)
    model = config_get[architecture](**init_params)
    return model


def mymodel_contain(architecture):
    return True if architecture in config_get else False


config_get = {
    # "HRNet": HRNet,
    # "ChangeModel": ChangeModel,
    # "ChangeModel_simple_dual": ChangeModel_simple_dual,
    # "ChangeModelNew": ChangeModelNew,
    # "ChangeModel_bifpn": ChangeModel_bifpn,
    # "cpnet": cpnet,
    "Unet_bifpn": Unet_bifpn,
    "EPUnet": EPUnet,
    "SGEPUnet": SGEPUnet,
    "UnetEdge": UnetEdge,
    # "Unet_swin": Unet_swin
}


class SegmentationScale(nn.Module):
    def __init__(
            self,
            model: nn.Module, 
            scale: float
        ):
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, x):
        oldsize = x.shape[-1]
        x = F.interpolate(x, scale_factor=self.scale)
        x = self.model(x)
        x = F.interpolate(x, size=[oldsize, oldsize], mode='bilinear')
        return x
    