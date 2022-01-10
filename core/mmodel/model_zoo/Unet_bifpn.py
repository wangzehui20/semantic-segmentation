import torch
from typing import Optional, Union, List
from segmentation_models_pytorch.encoders import get_encoder
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from segmentation_models_pytorch.unet.decoder import UnetDecoder, CenterBlock, DecoderBlock
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.base import SegmentationHead
from .Bifpn import BiFPN
# from Bifpn import BiFPN


class SegmentUnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        # encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):

        # features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class SegmentBifpnDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            num_channels=128,
            classnum=7
    ):
        super().__init__()
        self.decoder = BiFPN(
            num_channels=num_channels,
            conv_channels=encoder_channels,
            first_time=True,
            classnum=classnum
        )

    def forward(self, features):
        x, features = self.decoder(features)
        return x, features


class Unet_bifpn(torch.nn.Module):
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: str = "imagenet",
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            in_channels: int = 3,
            classes: int = 1
    ):
        super().__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        # Bifpn
        self.segmentbifpn_decoder = SegmentBifpnDecoder(
            encoder_channels=self.encoder.out_channels,
            classnum=decoder_channels[-1],
        )
        # Unet decoder
        self.decoder = SegmentUnetDecoder(
            encoder_channels=[128, 128, 128, 128, 128],
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type="scse",
        )

        self.segment_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )

        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.segmentbifpn_decoder)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features0 = self.encoder(x)
        
        _, features0 = self.segmentbifpn_decoder(features0)
        features0 = self.decoder(features0)
        segment_out0 = self.segment_head(features0)
        return segment_out0


if __name__ == '__main__':
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU')

    device = torch.device("cuda:0" if train_on_gpu else "cpu")

    ENCODER = "efficientnet-b1"
    ENCODER_WEIGHTS = None #"imagenet"
    ACTIVATION = None

    model = Unet_bifpn(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
    )

    img = np.zeros((1, 3, 512, 512), 'float32')
    img = torch.from_numpy(img)
    img = img.to(device)
    model.to(device)
    segment_out0 = model(img)
    print(segment_out0.shape)
