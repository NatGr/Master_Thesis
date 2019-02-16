import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.checkpoint as cp

import torchvision.models as models

import sys
import math


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x * 0


class ZeroMake(nn.Module):
    def __init__(self, channels, spatial):
        super(ZeroMake, self).__init__()
        self.spatial = spatial
        self.channels = channels

    def forward(self, x):
        return torch.zeros([x.size()[0], self.channels, x.size()[2] // self.spatial, x.size()[3] // self.spatial],
                           dtype=x.dtype, layout=x.layout, device=x.device)


class MaskBlock(nn.Module):
    """A basic densenet bottled block with a mask to handle pruning"""
    def __init__(self, n_channels, growth_rate, efficient):
        super(MaskBlock, self).__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels, inter_channels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3,
                               padding=1, bias=False)

        self.activation = Identity()
        self.activation.register_backward_hook(self._fisher)
        self.register_buffer('mask', None)

        self.input_shape = None
        self.output_shape = None
        self.flops = None
        self.params = None
        self.in_channels = n_channels
        self.out_channels = growth_rate
        self.stride = 1
        self.efficient = efficient

        # Fisher method is called on backward passes
        self.running_fisher = 0

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, F.relu, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            out = cp.checkpoint(bn_function, *prev_features)
        else:
            out = bn_function(*prev_features)
        out = F.relu(self.bn2(out))
        if self.mask is not None:
            out = out * self.mask[None, :, None, None]
        else:
            self._create_mask(x, out)
        out = self.activation(out)
        self.act = out

        out = self.conv2(out)
        return out

    def _create_mask(self, x, out):
        """This takes an activation to generate the exact mask required. It also records input and output shapes
        for posterity."""
        self.mask = x.new_ones(out.shape[1])
        self.input_shape = x.size()
        self.output_shape = out.size()

    def _fisher(self, _, __, grad_output):
        act = self.act.detach()
        grad = grad_output[0].detach()

        g_nk = (act * grad).sum(-1).sum(-1)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        self.running_fisher += del_k

    def reset_fisher(self):
        self.running_fisher = 0 * self.running_fisher

    def update(self, previous_mask):
        # This is only required for non-modular nets.
        return None

    def cost(self):

        in_channels = self.in_channels
        out_channels = self.out_channels
        middle_channels = int(self.mask.sum().item())

        conv1_size = self.conv1.weight.size()
        conv2_size = self.conv2.weight.size()

        self.params = in_channels * middle_channels * conv1_size[2] * conv1_size[3] + middle_channels * out_channels * \
                      conv2_size[2] * conv2_size[3]

        self.params += 2 * in_channels + 2 * middle_channels

    def compress_weights(self):
        """
        removes the pruned channels definitively rather than hiding them behind self.mask
        returns the number of channels left and the initial number of channels in the layer
        """
        middle_dim = int(self.mask.sum().item())
        full_middle_dim = self.mask.size(0)
        print(f"{middle_dim} channels left out of {full_middle_dim}")

        if middle_dim is not 0:
            conv1 = nn.Conv2d(self.in_channels, middle_dim, kernel_size=3, stride=1, bias=False)
            conv1.weight = nn.Parameter(self.conv1.weight[self.mask == 1, :, :, :])

            # Batch norm 2 changes
            bn2 = nn.BatchNorm2d(middle_dim)
            bn2.weight = nn.Parameter(self.bn2.weight[self.mask == 1])
            bn2.bias = nn.Parameter(self.bn2.bias[self.mask == 1])
            bn2.running_mean = self.bn2.running_mean[self.mask == 1]
            bn2.running_var = self.bn2.running_var[self.mask == 1]

            conv2 = nn.Conv2d(middle_dim, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            conv2.weight = nn.Parameter(self.conv2.weight[:, self.mask == 1, :, :])

        if middle_dim is 0:
            conv1 = Zero()
            bn2 = Zero()
            conv2 = ZeroMake(channels=self.out_channels, spatial=self.stride)

        self.conv1 = conv1
        self.conv2 = conv2
        self.bn2 = bn2

        if middle_dim is not 0:
            self.mask = torch.ones(middle_dim)
        else:
            self.mask = torch.ones(1)

        return middle_dim, full_middle_dim


class Bottleneck(nn.Module):
    """Desenet layers with bottlenecks of fixed size"""
    def __init__(self, n_channels, growth_rate, efficient, width=1):
        super(Bottleneck, self).__init__()
        inter_channels = int(4 * growth_rate * width)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels, inter_channels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3,
                               padding=1, bias=False)
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.bn1, F.relu, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            out = cp.checkpoint(bn_function, *prev_features)
        else:
            out = bn_function(*prev_features)
        out = self.conv2(F.relu(self.bn2(out)))
        return out


class SingleLayer(nn.Module, efficient):
    """DenseNet layer without bottlenecks"""
    def __init__(self, n_channels, growth_rate, efficient):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels, growth_rate, kernel_size=3,
                               padding=1, bias=False)
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.bn1, F.relu, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            out = cp.checkpoint(bn_function, *prev_features)
        else:
            out = bn_function(*prev_features)
        return out


class Transition(nn.Module):
    """
    layer of transition between two different dense blocks (nbr_channels reduced by compression rate),
    the width and height of feature maps will be divided by two
    """
    def __init__(self, n_channels, n_out_channels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(n_channels, n_out_channels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    """ The DenseNet was modified to consume less memory (at the cost of training speed), this can be disabled by setting
    efficient to False, see https://github.com/gpleiss/efficient_densenet_pytorch """
    def __init__(self, growth_rate, depth, reduction, n_classes, bottleneck, mask=False, width=1, efficient=True):
        super(DenseNet, self).__init__()

        n_dense_blocks = (depth - 4) // 3
        if bottleneck:
            n_dense_blocks //= 2

        n_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, n_channels, kernel_size=3, padding=1,
                               bias=False)

        self.dense1 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, mask, width)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = Transition(n_channels, n_out_channels)

        n_channels = n_out_channels
        self.dense2 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, mask, width)
        n_channels += n_dense_blocks * growth_rate
        n_out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = Transition(n_channels, n_out_channels)

        n_channels = n_out_channels
        self.dense3 = self._make_dense(n_channels, growth_rate, n_dense_blocks, bottleneck, mask, width)
        n_channels += n_dense_blocks * growth_rate

        self.bn1 = nn.BatchNorm2d(n_channels)
        self.fc = nn.Linear(n_channels, n_classes)
        self.efficient = efficient

        # Count params that don't exist in blocks (conv1, bn1, fc, trans1, trans2, trans3)
        self.fixed_params = len(self.conv1.weight.view(-1)) + len(self.bn1.weight) + len(self.bn1.bias) + \
                            len(self.fc.weight.view(-1)) + len(self.fc.bias)
        self.fixed_params += len(self.trans1.conv1.weight.view(-1)) + 2 * len(self.trans1.bn1.weight)
        self.fixed_params += len(self.trans2.conv1.weight.view(-1)) + 2 * len(self.trans2.bn1.weight)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, n_channels, growth_rate, n_dense_blocks, bottleneck, mask=False, width=1):
        layers = []
        for i in range(int(n_dense_blocks)):
            if bottleneck and mask:
                layers.append(MaskBlock(n_channels, growth_rate, self.efficient))
            elif bottleneck:
                layers.append(Bottleneck(n_channels, growth_rate, self.efficient, width))
            else:
                layers.append(SingleLayer(n_channels, growth_rate, self.efficient))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    @staticmethod
    def forward_dense(dense, init_features):
        features = [init_features]
        for name, layer in dense.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.forward_dense(self.dense1, out))
        out = self.trans2(self.forward_dense(self.dense2, out))
        out = self.forward_dense(self.dense3, out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = self.fc(out)
        return out
