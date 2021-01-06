import math
import torch.nn as nn
import torch.nn.functional as F

class StaticSamePadding(nn.Module):
    def __init__(self, layer, *args, **kwargs):
        super().__init__()
        self.layer = layer(*args, **kwargs)

        self.stride = self.get_list_val(self.layer.stride)
        self.kernel_size = self.get_list_val(self.layer.kernel_size)

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.layer(x)
        return x

    def get_list_val(self, val):
        if isinstance(val, int):
            val = [val] * 2
        elif len(val) == 1:
            val = [val[0]] * 2
        return val