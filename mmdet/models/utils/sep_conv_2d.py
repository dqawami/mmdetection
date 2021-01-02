import torch.nn as nn

from mmcv.cnn import normal_init, ConvModule
from .conv_bn import ConvBn

class SeparableConv2d(ConvBn):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=1, **kwargs)
        self.depth_wise = ConvModule(in_channels, in_channels, kernel_size=kernel_size,
                                     stride=1, padding=1, dilation=1, groups=in_channels,
                                     act_cfg=None)

    def forward(self, x):
        out = self.depth_wise(x)
        out = super().forward(out)
        return out

    def init_weights(self):
        normal_init(self.depth_wise, std=0.01)
        super().init_weights()
