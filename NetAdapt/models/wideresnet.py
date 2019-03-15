"""
Wideresnet class to be pruned by NetAdapt
@Author: Nathan Greffe
"""

import math
import torch
import torch.nn as nn


class WideResNet(nn.Module):
    """whole WideResNet module without bottlenecks"""
    def __init__(self, depth, widen_factor, num_classes=10, drop_rate=0.0):
        super(WideResNet, self).__init__()

        n_channels = [16, int(16 * widen_factor), int(32 * widen_factor), int(64 * widen_factor)]
        assert ((depth - 4) % 6 == 0)  # 4 = the initial conv layer + the 3 conv1*1 when we change the width
        n = (depth - 4) / 6
