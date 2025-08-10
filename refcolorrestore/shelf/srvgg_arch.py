from torch import nn as nn
import torch
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, _, out_lr):
        x = out_lr
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        out += F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        return F.interpolate(out, size=_.shape[2:], mode='bilinear', align_corners=False).clamp(0, 1.)


class RestorationVGGNetCompact(SRVGGNetCompact):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, upscale=1, **kwargs)

    def forward(self, x, _):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        out += x
        return out.clamp(0, 1.)


class DualVGGNetCompact(SRVGGNetCompact):
    def __init__(self, num_in_ch_color=3, num_in_ch_gray=3, *args, **kwargs):
        super().__init__(num_in_ch=num_in_ch_color+num_in_ch_gray, upscale=1, *args, **kwargs)

    def forward(self, x, lr):
        base = F.interpolate(lr, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([base, x], dim=1)
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        out += base
        return out.clamp(0, 1.)


####################################
# RETURN INITIALIZED MODEL INSTANCES
####################################


def srvgg(
        feature_channels: int = 24,
        num_blocks: int = 4,
        scale: int = 4,
):
    model = SRVGGNetCompact(num_in_ch=3,
                            num_out_ch=3,
                            num_feat=feature_channels,
                            num_conv=num_blocks,
                            upscale=scale)
    return model


def restorevgg(
        feature_channels: int = 24,
        num_blocks: int = 4,
):
    model = RestorationVGGNetCompact(num_in_ch=3,
                                     num_out_ch=3,
                                     num_feat=feature_channels,
                                     num_conv=num_blocks)
    return model


def dualvgg(
        feature_channels: int = 24,
        num_blocks: int = 4,
):
    model = DualVGGNetCompact(num_in_ch_color=3,
                              num_in_ch_gray=3,
                              num_out_ch=3,
                              num_feat=feature_channels,
                              num_conv=num_blocks)
    return model
