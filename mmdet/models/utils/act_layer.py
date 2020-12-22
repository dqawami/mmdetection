import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

act_fn_list = ["silu", "swish", "hswish", "relu", "relu6", "mish", "srelu"]

class ActLayer(nn.Module):
    def __init__(self, act_name):
        super(ActLayer, self).__init__()
        assert act_name in act_fn_list
        self.act_fn = act_name

    def forward(self, nodes):
        # Activation function
        if (self.act_fn == "silu"):
            nodes = nodes * torch.sigmoid(nodes)

        # # Quantization-friendly hard swish
        elif (self.act_fn == "swish"):
            nodes = nodes * F.relu6(nodes + 3) / 6
            nodes = nodes * torch.sigmoid(nodes)

        elif (self.act_fn == "hswish"):
            nodes = nodes * F.relu6(nodes + 3) / 6

        elif (self.act_fn == "relu"):
            nodes = F.relu(nodes)

        elif (self.act_fn == "relu6"):
            nodes = F.relu6(nodes)

        elif (self.act_fn == "mish"):
            nodes = nodes * F.tanh(F.softplus(nodes))

        elif (self.act_fn == "srelu"):
            beta = numpy.array([20.0])
            beta = torch.autograd.Variable(torch.from_numpy(beta)) ** 2
            beta = (beta ** 2).type(nodes.type())
            safe_log = torch.log(torch.where(nodes > 0., beta * nodes + 1., torch.ones_like(nodes)))
            nodes = torch.where((nodes > 0.), nodes - (1. / beta) * safe_log, torch.zeros_like(nodes))

        return nodes