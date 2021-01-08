# Based off of github.com/rwightman/efficientdet-pytorch

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, ConvModule
from mmdet.core import auto_fp16

from ..builder import NECKS
from ..utils import ActLayer, SeparableConv2d

class BiFPNNode(nn.Module):
    def __init__(self, input_channels, output_channel, num_backbone_features,
                 weight_method, act_fn, separable_conv, epsilon, input_offsets,
                 target_reduction, reduction):
        super().__init__()

        self.weight_method = weight_method
        self.act_layer = ActLayer(act_fn)
        self.epsilon = epsilon

        self.input_layer = nn.ModuleDict()

        self.offsets = input_offsets

        for offset in input_offsets:
            offset_nodes = nn.Sequential()
            used_input = output_channel
            if offset < num_backbone_features:
                used_input = input_channels
            input_reduction = reduction[offset]

            reduction_ratio = target_reduction / input_reduction

            if used_input != output_channel:
                conv = ConvModule(used_input, output_channel, kernel_size=1,
                                  norm_cfg=dict(type='BN'), act_cfg=None)
                offset_nodes.add_module("conv", conv)

            if reduction_ratio > 1:
                stride_size = int(reduction_ratio)
                offset_nodes.add_module("max_pool", nn.MaxPool2d(
                    kernel_size=stride_size + 1, stride=stride_size, padding=1
                ))

            elif reduction_ratio < 1:
                scale = int(1 // reduction_ratio)
                offset_nodes.add_module("upsample", nn.UpsamplingNearest2d(
                    scale_factor=scale))

            self.input_layer[str(offset)] = offset_nodes

        if self.weight_method != "sum":
            self.edge_weights = nn.Parameter(torch.ones(len(input_offsets)), requires_grad=True)

        conv_kwargs = dict(in_channels=output_channel, out_channels=output_channel, kernel_size=3,
                           act_cfg=None, norm_cfg=dict(type='BN'))

        if separable_conv:
            self.fusion_convs = SeparableConv2d(**conv_kwargs)
        else:
            self.fusion_convs = ConvModule(padding=1, **conv_kwargs)

    def forward(self, inputs):
        # Create node inputs
        nodes = []
        dtype = inputs[0].dtype

        for offset in self.offsets:
            nodes.append(self.input_layer[str(offset)](inputs[offset]))

        for i in range(1, len(nodes)):
            if nodes[0].size() != nodes[i].size():
                nodes[i] = F.interpolate(nodes[i], size=(nodes[0].size(2), nodes[0].size(3)))

        # Weight method
        # Softmax normalized fusion
        if self.weight_method == "attn":
            normalized_weights = torch.softmax(self.edge_weights.type(dtype), dim=0)
            nodes = torch.stack(nodes, dim=-1) * normalized_weights

        # Fast normalized feature fusion
        elif self.weight_method == "fast_attn":
            edge_weights = F.relu(self.edge_weights.type(dtype))
            weights_sum = torch.sum(edge_weights)

            nodes = torch.stack(
                [(nodes[i] * edge_weights[i]) / (weights_sum + self.epsilon)
                 for i in range(len(nodes))], dim=-1)

        elif self.weight_method == "sum":
            nodes = torch.stack(nodes, dim=-1)

        nodes = torch.sum(nodes, dim=-1)

        nodes = self.fusion_convs(nodes)

        return self.act_layer(nodes)


class BiFPNBlock(nn.Module):

    def __init__(self, input_channels, num_backbone_features, num_outs,
                 channels, weight_method, act_fn, separable_conv, epsilon,
                 input_offsets, reduction):
        super().__init__()

        weight_method_list = ["attn", "fast_attn", "sum"]
        self.num_outs = num_outs

        assert self.num_outs >= 3
        assert weight_method in weight_method_list

        self.nodes = nn.ModuleList()

        for i in range(num_outs - 1):
            self.nodes.append(BiFPNNode(input_channels[num_outs - i - 2],
                                        channels,
                                        num_backbone_features,
                                        weight_method,
                                        act_fn,
                                        separable_conv,
                                        epsilon,
                                        input_offsets[i],
                                        reduction[num_outs + i],
                                        reduction))

        for i in range(1, num_outs):
            self.nodes.append(BiFPNNode(input_channels[i],
                                        channels,
                                        num_backbone_features,
                                        weight_method,
                                        act_fn,
                                        separable_conv,
                                        epsilon,
                                        input_offsets[num_outs + i - 2],
                                        reduction[num_outs * 2 + i - 2],
                                        reduction))

    def forward(self, inputs):
        output = list(inputs)

        for node in self.nodes:
            output.append(node(output))

        return tuple(output[-self.num_outs::])


@NECKS.register_module
class BiFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 input_indices,
                 num_layers,
                 num_outs,
                 strides,
                 weight_method="fast_attn",
                 act_cfg="silu",
                 separable_conv=True,
                 epsilon=0.0001,
                 reduction_ratio=2.0):
        super(BiFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.num_backbone_features = len(in_channels)

        assert self.num_backbone_features >= 2
        assert num_outs - self.num_backbone_features >= 2
        assert len(input_indices) == len(strides)

        # Check that input_indices are dense
        for i in range(len(input_indices) - 1):
            assert input_indices[i] + 1 == input_indices[i + 1]

        self.in_channels = in_channels
        self.num_outs = num_outs

        self.extra_convs = nn.ModuleList()

        min_level = input_indices[0]
        max_level = num_outs + min_level - 1

        # input_offsets are the nodes that the current node is getting its inputs from (either 2 or 3).
        input_offsets = []
        # reduction is the reduction values for the inputted and neck nodes
        reduction = strides

        for i in range(self.num_outs - self.num_backbone_features):
            if i == 0:
                input_channels = in_channels[-1]
            else:
                input_channels = out_channels
            if input_channels != out_channels:
                self.extra_convs.append(
                    nn.Sequential(
                        ConvModule(input_channels, out_channels, kernel_size=1,
                                   norm_cfg=dict(type='BN'), act_cfg=None),
                        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                    )
                )
            else:
                self.extra_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            reduction.append(int(reduction[-1] * reduction_ratio))

        node_ids = {min_level + i: [i] for i in range(num_outs)}

        level_last_id = lambda level: node_ids[level][-1]
        level_all_ids = lambda level: node_ids[level]
        id_cnt = itertools.count(num_outs)

        for i in range(max_level - 1, min_level - 1, -1):
            # top-down path
            reduction.append(reduction[i - min_level])
            input_offsets.append([level_last_id(i), level_last_id(i + 1)])
            node_ids[i].append(next(id_cnt))

        for i in range(min_level + 1, max_level + 1):
            # bottom-up path
            reduction.append(reduction[i - min_level])
            input_offsets.append(level_all_ids(i) + [level_last_id(i - 1)])
            node_ids[i].append(next(id_cnt))

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_channels = in_channels + [out_channels, ] * (self.num_outs - self.num_backbone_features)
            else:
                input_channels = [out_channels, ] * self.num_outs
            self.layers.append(
                BiFPNBlock(input_channels, self.num_backbone_features, self.num_outs, out_channels, weight_method,
                           act_cfg, separable_conv, epsilon, input_offsets, reduction)
            )

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        extra_inputs = []
        for i in range(self.num_outs - len(self.in_channels)):
            if i == 0:
                extra_inputs.append(self.extra_convs[i](inputs[-1]))
            else:
                extra_inputs.append(self.extra_convs[i](extra_inputs[-1]))

        outputs = inputs + extra_inputs
        for layer in self.layers:
            outputs = layer(outputs)

        return tuple(outputs)

