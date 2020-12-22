import torch.nn as nn

from mmcv.cnn import normal_init, ConvModule

# temp until DepthwsieSeparableConvModule can be imported
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.depth_wise = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, kernel_size),
                                    stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=in_channels)
        self.point_wise = ConvModule(in_channels, out_channels, kernel_size=1, **kwargs)

    def forward(self, x):
        out = self.depth_wise(x)
        out = self.point_wise(out)
        return out

    def init_weights(self):
        normal_init(self.depth_wise, std=0.01)
        normal_init(self.point_wise, std=0.01)
