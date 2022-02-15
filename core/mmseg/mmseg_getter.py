import torch.nn as nn
import torch
import os.path as osp
import torch.nn.functional as F
from mmcv.utils import Config as MMConfig
from mmseg.models import build_segmentor
from mmseg.ops import resize


class Activation(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softamx/logsoftmax/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)


class Segmodel(nn.Module):
    def __init__(
            self,
            architecture: str,
            in_channels: int = 3,
            classes: int = 1,
            ispretrain: bool = True,
            pretrained="",
            activation=None,
            with_auxiliary_head: bool = False
    ):
        super().__init__()
        self.activation = Activation(activation)

        cfg = MMConfig.fromfile(config_get[architecture]["config_path"])
        cfg.model.backbone.in_channels = in_channels
        if not ispretrain:
            cfg.model.pretrained = None
        if ispretrain and osp.exists(pretrained):
            cfg.model.pretrained = pretrained
        # multi decode head
        if isinstance(cfg.model.decode_head, list):
            for index in range(len(cfg.model.decode_head)):
                cfg.model.decode_head[index].num_classes = classes
        else:
            cfg.model.decode_head.num_classes = classes

        # auxiliary decode head
        self.with_auxiliary_head = with_auxiliary_head
        if self.with_auxiliary_head:
            cfg.model.auxiliary_head.num_classes = classes
            self.auxiliary_head_loss_weight = cfg.model.auxiliary_head.loss_decode.loss_weight

        self.model = build_segmentor(cfg.model)
        if cfg.model.pretrained:
            self.model.init_weights()

    def _forward_dummy_aux(self, image: torch.Tensor):
        x = self.model.extract_feat(image)
        out = self._auxiliary_head_forward_test(x, None)
        out = resize(
            input=out,
            size=image.shape[2:],
            mode='bilinear',
            align_corners=self.model.align_corners)
        return out

    def _auxiliary_head_forward_test(self, x, img_metas):
        seg_logits = self.model.auxiliary_head.forward_test(x, img_metas, self.model.test_cfg)
        return seg_logits

    def forward(self, image: torch.Tensor):
        if self.with_auxiliary_head:
            return self.activation(self.model.forward_dummy(image)), self.activation(self._forward_dummy_aux(image))
        else:
            return self.activation(self.model.forward_dummy(image))


def get_mmseg_model(architecture, **kwargs):
    if architecture in config_get:
        return Segmodel(architecture, **kwargs)


def mmseg_contain(architecture):
    return True if architecture in config_get else False


config_get = {
    "ocrnet_hr18": {
        "config_path": "core/mmseg/config/ocrnet_hr18.py"
    },
    "segformer_b0": {
        "config_path": "/data/code/semantic-segmentation-semi-supervised-learning/core/mmseg/config/segformer_mit-b0.py"
    },
    "segformer_b2": {
        "config_path": "/data/code/semantic-segmentation-semi-supervised-learning/core/mmseg/config/segformer_mit-b2.py"
    },
    "segformer_b3": {
        "config_path": "core/mmseg/config/segformer_mit-b3.py"
    },
    "segformer_b4": {
        "config_path": "core/mmseg/config/segformer_mit-b4.py"
    },
    "upernet_swin-s": {
        "config_path": "/data/code/semantic-segmentation-semi-supervised-learning/core/mmseg/config/upernet_swin-s_c96.py"
    }
}
