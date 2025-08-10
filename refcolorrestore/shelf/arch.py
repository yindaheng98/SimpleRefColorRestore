import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .modules import *


####################################
# DEFINE MODEL INSTANCES
####################################


class RT4KSR_Rep(nn.Module):
    def __init__(self,
                 num_channels_in,
                 num_channels_out,
                 num_feats,
                 num_blocks,
                 upscale,
                 act,
                 eca_gamma,
                 is_train,
                 forget,
                 layernorm,
                 residual) -> None:
        super().__init__()
        self.forget = forget
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gaussian = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=1)
        self.upscale = upscale

        self.down = nn.PixelUnshuffle(2)
        self.up = nn.PixelShuffle(2)
        self.head = nn.Sequential(nn.Conv2d(num_channels_in * (2**2), num_feats, 3, padding=1))

        hfb = []
        if is_train:
            hfb.append(ResBlock(num_feats, ratio=2))
        else:
            hfb.append((RepResBlock(num_feats)))
        hfb.append(act)
        self.hfb = nn.Sequential(*hfb)

        body = []
        for i in range(num_blocks):
            if is_train:
                body.append(SimplifiedNAFBlock(in_c=num_feats, act=act, exp=2, eca_gamma=eca_gamma, layernorm=layernorm, residual=residual))
            else:
                body.append(SimplifiedRepNAFBlock(in_c=num_feats, act=act, exp=2, eca_gamma=eca_gamma, layernorm=layernorm, residual=residual))

        self.body = nn.Sequential(*body)

        tail = [LayerNorm2d(num_feats)]
        if is_train:
            tail.append(ResBlock(num_feats, ratio=2))
        else:
            tail.append(RepResBlock(num_feats))
        self.tail = nn.Sequential(*tail)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_feats, num_channels_out * ((2 * upscale) ** 2), 3, padding=1),
            nn.PixelShuffle(upscale*2)
        )

    def forward(self, _, out_lr):  # only lr is used
        x = out_lr
        # stage 1
        hf = x - self.gaussian(x)

        # unshuffle to save computation
        x_unsh = self.down(x)
        hf_unsh = self.down(hf)

        shallow_feats_hf = self.head(hf_unsh)
        shallow_feats_lr = self.head(x_unsh)

        # stage 2
        deep_feats = self.body(shallow_feats_lr)
        hf_feats = self.hfb(shallow_feats_hf)

        # stage 3
        if self.forget:
            deep_feats = self.tail(self.gamma * deep_feats + hf_feats)
        else:
            deep_feats = self.tail(deep_feats)

        out = self.upsample(deep_feats)
        out += F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        return F.interpolate(out, size=_.shape[2:], mode='bilinear', align_corners=False).clamp(0, 1.)


class RT4KRestore_Rep(RT4KSR_Rep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, upscale=1, **kwargs)

    def forward(self, x, _):  # only lr is used
        shape = x.shape[-2:]
        x = F.interpolate(x, size=(x.shape[-2]//2*2, x.shape[-1]//2*2), mode='bilinear', align_corners=False)
        # stage 1
        hf = x - self.gaussian(x)

        # unshuffle to save computation
        x_unsh = self.down(x)
        hf_unsh = self.down(hf)

        shallow_feats_hf = self.head(hf_unsh)
        shallow_feats_lr = self.head(x_unsh)

        # stage 2
        deep_feats = self.body(shallow_feats_lr)
        hf_feats = self.hfb(shallow_feats_hf)

        # stage 3
        if self.forget:
            deep_feats = self.tail(self.gamma * deep_feats + hf_feats)
        else:
            deep_feats = self.tail(deep_feats)

        out = self.upsample(deep_feats)
        out += x
        return F.interpolate(out, size=shape, mode='bilinear', align_corners=False).clamp(0, 1.)

####################################
# RETURN INITIALIZED MODEL INSTANCES
####################################


def rt4ksr_rep(
        act_type: str = "gelu",
        feature_channels: int = 24,
        num_blocks: int = 4,
        scale: int = 4,
        is_train: bool = True,
):
    act = activation(act_type)
    model = RT4KSR_Rep(num_channels_in=3,
                       num_channels_out=3,
                       num_feats=feature_channels,
                       num_blocks=num_blocks,
                       upscale=scale,
                       act=act,
                       eca_gamma=0,
                       forget=False,
                       is_train=is_train,
                       layernorm=True,
                       residual=True)
    return model


def rt4krestore_rep(
        act_type: str = "gelu",
        feature_channels: int = 24,
        num_blocks: int = 4,
        is_train: bool = True,
):
    act = activation(act_type)
    model = RT4KRestore_Rep(num_channels_in=3,
                            num_channels_out=3,
                            num_feats=feature_channels,
                            num_blocks=num_blocks,
                            act=act,
                            eca_gamma=0,
                            forget=False,
                            is_train=is_train,
                            layernorm=True,
                            residual=True)
    return model
