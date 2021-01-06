import torch.nn as nn

from .static_same_padding import StaticSamePadding
from mmcv.cnn import normal_init

class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, apply_bn=True, **kwargs):

        super().__init__()

        self.apply_bn = apply_bn

        self.point_wise = self.point_wise = StaticSamePadding(nn.Conv2d, in_channels, out_channels,
                                                              kernel_size, **kwargs)
        if apply_bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        out = self.point_wise(x)
        if self.apply_bn:
            out = self.bn(out)
        return out

    def init_weights(self):
        normal_init(self.point_wise, std=0.01)