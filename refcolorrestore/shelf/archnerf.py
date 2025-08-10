import torch
import torch.nn.functional as F

from .arch import *


class RT4KDual_Rep(RT4KSR_Rep):
    def __init__(self, num_channels_color, num_channels_gray, *args, **kwargs):
        super().__init__(*args, **kwargs, upscale=1, num_channels_in=num_channels_color + num_channels_gray)
        self.num_channels_color = num_channels_color
        self.num_channels_gray = num_channels_gray

    def forward(self, x, lr):
        shape = x.shape[-2:]
        x = F.interpolate(x, size=(x.shape[-2]//2*2, x.shape[-1]//2*2), mode='bilinear', align_corners=False)
        base = F.interpolate(lr, size=(x.shape[-2]//2*2, x.shape[-1]//2*2), mode='bicubic', align_corners=False)
        x = torch.cat([base, x], dim=1)

        # Following is just copy from RT4KSR_Rep
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
        out += base
        return F.interpolate(out, size=shape, mode='bilinear', align_corners=False).clamp(0, 1.)

####################################
# RETURN INITIALIZED MODEL INSTANCES
####################################


def rt4kdual_rep(
        act_type: str = "gelu",
        feature_channels: int = 24,
        num_blocks: int = 4,
        is_train: bool = True,
):
    act = activation(act_type)
    model = RT4KDual_Rep(num_channels_color=3,
                         num_channels_gray=3,
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
