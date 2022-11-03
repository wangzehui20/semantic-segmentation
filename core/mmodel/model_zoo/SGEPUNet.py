import torch
import torch.nn.functional as F
import numpy as np
from core.mmodel.model_zoo.backbones.resnet import BasicBlock
from torch import nn
from segmentation_models_pytorch.encoders import get_encoder
from torchvision import models


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels=512,
                 n_filters=256,
                 kernel_size=3,
                 is_deconv=False,
                 ):
        super().__init__()

        if kernel_size == 3:
            conv_padding = 1
        elif kernel_size == 1:
            conv_padding = 0

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels,
                               in_channels // 4,
                               kernel_size,
                               padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.LeakyReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4,
                                              in_channels // 4,
                                              3,
                                              stride=2,
                                              padding=1,
                                              output_padding=conv_padding, bias=False)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2)

        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.LeakyReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4,
                               n_filters,
                               kernel_size,
                               padding=conv_padding, bias=False)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.functional.leaky_relu
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Attetion(nn.Module):
    def __init__(self, inchannels, kernel_size=7):
        super(Attetion, self).__init__()
        self.CA = ChannelAttention(in_planes=inchannels)
        self.SA = SpatialAttention(kernel_size=kernel_size)

    def forward(self, input):
        out = self.CA(input) * input
        out = self.SA(out)
        return out


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


class PSPModule(nn.Module):
    def __init__(self, features, out_features=64, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class EPUnet(nn.Module):
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: str = "imagenet",
                 in_channels: int = 3,
                 classes: int = 1,
                 activation: str = None,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 is_edge=False,
                 ):
        super().__init__()
        filters = [64, 128, 256, 512]
        self.activation = Activation(activation)
        self.is_edge = is_edge
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        self.psp = PSPModule(8 + 64)
        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.firstbn = self.encoder.bn1
        self.firstrelu = self.encoder.relu
        self.firstmaxpool = self.encoder.maxpool
        self.encoder1 = self.encoder.layer1
        self.encoder2 = self.encoder.layer2
        self.encoder3 = self.encoder.layer3
        self.encoder4 = self.encoder.layer4

        # Decoder
        self.center = DecoderBlock(in_channels=filters[3],
                                   n_filters=filters[3],
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)
        self.decoder4 = DecoderBlock(in_channels=filters[3] + filters[2],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.attention4 = Attetion(filters[2])
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.attention3 = Attetion(filters[1])
        self.decoder2 = DecoderBlock(in_channels=filters[1] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.attention2 = Attetion(filters[0])
        self.decoder1 = DecoderBlock(in_channels=filters[0] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.attention1 = Attetion(filters[0])
        self.finalconv = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(32, classes, 1))
        self.midconv = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(32, classes, 1))
        self.res4 = BasicBlock(64, 64, stride=1, downsample=None)
        self.conv4 = nn.Conv2d(64, 32, 1)
        self.res3 = BasicBlock(32, 32, stride=1, downsample=None)
        self.conv3 = nn.Conv2d(32, 16, 1)
        self.res2 = BasicBlock(16, 16, stride=1, downsample=None)
        self.conv2 = nn.Conv2d(16, 8, 1)
        self.res1 = BasicBlock(8, 8, stride=1, downsample=None)
        self.conv1 = nn.Conv2d(8, 1, 1)

    def forward(self, x, get_mid=False):
        inp_size = x.shape[-2:]
        # stem
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)

        x_ = self.firstmaxpool(x)

        # Encoder
        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)

        d4 = self.decoder4(torch.cat([center, e3], 1))
        edge = self.res4(x) * F.interpolate(self.attention4(d4), x.shape[-2:], mode='bilinear', align_corners=True)
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        edge = self.res3(self.conv4(edge)) * F.interpolate(self.attention3(d3), x.shape[-2:], mode='bilinear',
                                                           align_corners=True)
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        edge = self.res2(self.conv3(edge)) * F.interpolate(self.attention2(d2), x.shape[-2:], mode='bilinear',
                                                           align_corners=True)
        d1 = self.decoder1(torch.cat([d2, x], 1))
        edge = self.res1(self.conv2(edge)) * F.interpolate(self.attention1(d1), x.shape[-2:], mode='bilinear',
                                                           align_corners=True)
        edge_result = F.interpolate(self.conv1(edge), inp_size, mode='bilinear', align_corners=True)
        f = self.psp(torch.cat([d1, F.interpolate(edge, inp_size, mode='bilinear', align_corners=True)], 1))
        mid = f
        finalseg = self.finalconv(f)

        edge_result = self.activation(edge_result)
        finalseg = self.activation(finalseg)

        if get_mid:
            return [finalseg, edge_result, mid]  # mid is not activation
        else:
            if self.is_edge:
                return [finalseg, edge_result]
            else:
                return finalseg


class SGEPUnet(nn.Module):
    def __init__(self,
                 classes=1,
                 pretrainpath=None,
                 activation=None
                 ):
        super().__init__()
        self.pretrainpath = pretrainpath
        self.beforeUnet = EPUnet(activation=activation)
        self.afterUnet = EPUnet(activation=activation)
        self.disconv = nn.Sequential(nn.Conv2d(64 * 2, 64, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, classes, 1))

        self.initialize_weights()

    def initialize_weights(self):
        pth = torch.load(self.pretrainpath)
        net_state_dict = self.beforeUnet.state_dict()
        pretrained_dict = {k.replace('module.', ''): v for k, v in pth['state_dict'].items() if k.replace('module.', '') in net_state_dict}
        net_state_dict.update(pretrained_dict)
        self.beforeUnet.load_state_dict(net_state_dict)
        self.afterUnet.load_state_dict(net_state_dict)
        print('Initialize success!')

    def forward(self, x):
        """
        :param x:
            x[0]: before data
            x[1]: after data
        :return:
        """
        pre1 = self.beforeUnet(x[0], get_mid=True)
        pre2 = self.afterUnet(x[1], get_mid=True)
        chg = self.disconv(torch.cat((pre1[-1] - pre2[-1], pre2[-1] - pre1[-1]), 1))
        return [[pre1, pre2, chg]]

# if __name__ == "__main__":
    # img = np.random.rand(1, 3, 256, 256)
    # label = np.zeros((1, 1, 256, 256))
    # x = torch.tensor(img, dtype=torch.float32)
    # y = torch.tensor(label, dtype=torch.float32)
    # b = EPUnet()
    # result = b(x, y)
    # print('done')
    # a=SGEPUnet(1,None).forward(x,x,1)
