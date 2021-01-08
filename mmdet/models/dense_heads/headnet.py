# Based off of github.com/rwightman/efficientdet-pytorch

import torch
import torch.nn as nn

from ..builder import HEADS
from .anchor_head import AnchorHead
from mmcv.cnn import normal_init, xavier_init, ConvModule

from ..utils import ActLayer, SeparableConv2d


class HeadLayer(nn.Module):
    def __init__(self, num_classes, in_channels, box_class_repeats, num_levels,
                 aspect_len, act_fn="silu", separable_conv=True, num_scales=3):
        super(HeadLayer, self).__init__()

        self.num_anchors = aspect_len * num_scales
        self.num_classes = num_classes

        self.box_class_repeats = box_class_repeats
        self.num_levels = num_levels
        self.act_layer = ActLayer(act_fn)

        self.conv_rep = nn.ModuleList()
        self.bn_rep = nn.ModuleList()

        conv_kwargs = dict(in_channels=in_channels, out_channels=in_channels,
                           kernel_size=3, bias=False, act_cfg=None)

        for _ in range(box_class_repeats):
            if separable_conv:
                conv = SeparableConv2d(**conv_kwargs)
            else:
                conv = ConvModule(**conv_kwargs)
            self.conv_rep.append(conv)

            bn_levels = []
            for _ in range(num_levels):
                bn = nn.BatchNorm2d(in_channels)
                bn_levels.append(bn)
            self.bn_rep.append(nn.ModuleList(bn_levels))

        predict_kwargs = dict(in_channels=in_channels, out_channels=num_classes * self.num_anchors,
                              kernel_size=3, bias=True, act_cfg=None)

        if separable_conv:
            self.predict = SeparableConv2d(**predict_kwargs)
        else:
            self.predict = ConvModule(**predict_kwargs)

    def forward(self, feats):
        outputs = []
        for level in range(self.num_levels):
            feat_level = feats[level]
            for i in range(self.box_class_repeats):
                feat_level = self.conv_rep[i](feat_level)
                feat_level = self.bn_rep[i][level](feat_level)
                feat_level = self.act_layer(feat_level)

            feat_level = self.predict(feat_level)

            outputs.append(feat_level)
        return outputs


@HEADS.register_module()
class HeadNet(AnchorHead):
    def __init__(self, num_classes, in_channels, box_class_repeats, num_levels,
                 act_fn="silu", separable_conv=True, num_scales=3,
                 anchor_generator=dict(type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 **kwargs):

        self.levels = []
        for i in range(num_levels):
            self.levels.append(i)

        self.net_kwargs = dict(
            box_class_repeats=box_class_repeats,
            num_levels=num_levels, aspect_len=len(anchor_generator.ratios),
            act_fn=act_fn, separable_conv=separable_conv, num_scales=num_scales
        )

        super(HeadNet, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            **kwargs
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def _init_layers(self):
        self.cls_net = HeadLayer(self.cls_out_channels, self.in_channels, **self.net_kwargs)
        self.bbox_net = HeadLayer(4, self.in_channels, **self.net_kwargs)

    def forward(self, feats):
        cls_score = self.cls_net(feats)
        bbox_pred = self.bbox_net(feats)

        return cls_score, bbox_pred
