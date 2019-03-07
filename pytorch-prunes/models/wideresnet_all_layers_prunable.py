"""WideResNet architecture that is built from a dict of tensors, it copies the structure of wideresnet.py in the
morph_net folder
Author: Nathan Greffe"""

import torch
import torch.nn as nn
import math

class WideResNetBlock(nn.Module):
    """wideresnet block, """
    def __init__(self, is_first, in_channels, mid_channels, out_channels, stride, drop_rate):
        """is_first means that we will have a skip connection, mid_channels must be != 0"""
        super(WideResNetBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                      padding=0, bias=False) if is_first else None

    def forward(self, x):
        has_conv_shortcut = self.conv_shortcut is not None
        if has_conv_shortcut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(x if has_conv_shortcut else out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)

        return torch.add(self.conv_shortcut(x) if has_conv_shortcut else x, out)


class WideResNetSubNetwork(nn.Module):
    """subNetwork (several resnet blocks with the same number of channels) of WideResNet"""
    def __init__(self, block_id, nb_layers, in_channels, out_channels, stride, drop_rate, channels_dict):
        super(WideResNetSubNetwork, self).__init__()

        layers = []
        for i in range(int(nb_layers)):
            conv1_name = f"Conv_{block_id}_{i}_1"
            if conv1_name not in channels_dict or channels_dict[conv1_name] == 0:
                if i == 0:  # in the case the first layer was skipped, we have to use the skip 1*1 convolution as
                    # input for the second layer
                    layers.append(nn.BatchNorm2d(in_channels))
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))

                # otherwise there is nothing to add to the WRN
            else:
                layers.append(WideResNetBlock(i == 0, in_channels if i == 0 else out_channels,
                                              channels_dict[conv1_name], out_channels, stride if i == 0 else 1,
                                              drop_rate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetAllLayersPrunable(nn.Module):
    """whole WideResNet module whose every layer is prunable instead of just the bottleneck layers"""
    def __init__(self, depth, channels_dict, num_classes=10, drop_rate=0.0):
        super(WideResNetAllLayersPrunable, self).__init__()

        assert ((depth - 4) % 6 == 0)  # 4 = the initial conv layer + the 3 conv1*1 when we change the width
        n = (depth - 4) / 6

        first_layer_name = "Conv_0"

        n_channels = [channels_dict[first_layer_name], channels_dict["Skip_1"],
                      channels_dict["Skip_2"], channels_dict["Skip_3"]]

        # 1st conv before any network block
        self.Conv_0 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1,
                                padding=1, bias=False)
        # 1st block
        self.block1 = WideResNetSubNetwork(1, n, n_channels[0], n_channels[1], 1, drop_rate, channels_dict)
        # 2nd block
        self.block2 = WideResNetSubNetwork(2, n, n_channels[1], n_channels[2], 2, drop_rate, channels_dict)
        # 3rd block
        self.block3 = WideResNetSubNetwork(3, n, n_channels[2], n_channels[3], 2, drop_rate, channels_dict)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.nChannels = n_channels[3]

        # Count params that don't exist in blocks (conv1, bn1, fc)
        self.fixed_params = len(self.Conv_0.weight.view(-1)) + len(self.bn1.weight) + len(self.bn1.bias) + \
            len(self.fc.weight.view(-1)) + len(self.fc.bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.Conv_0(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avg_pool(out)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
