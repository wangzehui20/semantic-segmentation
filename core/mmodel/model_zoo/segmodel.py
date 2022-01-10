import torch
import torch.nn as nn
import os.path as osp
from .backbones import MixVisionTransformer
from .decode_heads import SegformerHead
from mmseg.ops import resize
from .encoder_decoder_config.encoder_config.mit import mit_b0_cfg
from .encoder_decoder_config.decoder_config import segformer_head_cfg


def get_encoder_cfg(encoder_name):
    if encoder_name in encoder_cfg:
        return encoder_cfg[encoder_name]
    else:
        raise ValueError(f'No {encoder_name} config in encoder configs')


def get_decoder_cfg(decoder_name):
    if decoder_name in decoder_cfg:
        return decoder_cfg[decoder_name]
    else:
        raise ValueError(f'No {decoder_name} config in decoder configs')


def get_encoder_model(encoder_name):
    if encoder_name in encoder_model:
        return encoder_model[encoder_name]
    else:
        raise ValueError(f'No {encoder_name} model in encoder models')


def get_decoder_model(decoder_name):
    if decoder_name in decoder_model:
        return decoder_model[decoder_name]
    else:
        raise ValueError(f'No {decoder_name} model in decoder models')


def get_neck_model(neck_name):
    if neck_name in neck_model:
        return neck_model[neck_name]
    else:
        raise ValueError(f'No {neck_name} model in neck models')


encoder_cfg = {
    'mit': mit_b0_cfg,
}

decoder_cfg = {
    'segformer_head': segformer_head_cfg,
}

encoder_model = {
    'mit': MixVisionTransformer,
}

decoder_model = {
    'segformer_head': SegformerHead,
}

neck_model = {
}


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


class segmodel(nn.Module):
    def __init__(self,
                 encoder_name: str,
                 decoder_name: str,
                 neck_name: str = None,
                 in_channels: int = 3,
                 classes: int = 1,
                 activation: str = None,
                 pretrain_path: str = ''
                 ):
        super(segmodel, self).__init__()
        encoder_model = get_encoder_model(encoder_name)
        decoder_model = get_decoder_model(decoder_name)
        encoder_cfg = get_encoder_cfg(encoder_name)
        decoder_cfg = get_decoder_cfg(decoder_name)
        encoder_cfg['in_channels'] = in_channels
        decoder_cfg['num_classes'] = classes
        self.encoder = encoder_model(**encoder_cfg)
        self.decoder = decoder_model(**decoder_cfg)

        if neck_name:
            self.neck = get_neck_model(neck_name)
        self.with_neck = True if neck_name else False
        self.align_corners = decoder_cfg['align_corners']
        self.activation = Activation(activation)

        if osp.exists(pretrain_path):
            self.pretrained = pretrain_path
        else:
            self.pretrained = None
        if self.pretrained:
            self.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.encoder(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def init_weights(self):
        state_dict = torch.load(self.pretrained, map_location=torch.device("cpu"))["state_dict"]
        self.encoder.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})

    def encode_decode(self, img):
        features = self.extract_feat(img)
        out = self.decoder(features)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def forward(self, x):
        return self.activation(self.encode_decoe(x))
