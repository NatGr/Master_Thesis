import math
import torch
import torch.nn as nn


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


class BasicBlock(nn.Module):
    """A basic resnet block withtout a mask nor a bottleneck"""
    def __init__(self, in_channels, out_channels, stride, droprate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = droprate
        self.equalInOut = (in_channels == out_channels)
        self.convShortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                                padding=0, bias=False) if not self.equalInOut else None
        # initially, they wrote
        # (not self.equalInOut) and nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
        #                                                 padding=0, bias=False) or None
        # in this case it is equivalent to a ternary since (a and b or c) == (b if a else c) iff b and c are not
        # booleans, I have no idea why they did not write a ternary expression in the first place

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class BottleBlock(nn.Module):
    """A basic resnet block withtout a mask but with a bottleneck (3*3 -> 3*3 not 1*1 -> 3*3)"""
    def __init__(self, in_channels, out_channels, mid_channels, stride, drop_rate=0.0):
        super(BottleBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = drop_rate
        self.equalInOut = (in_channels == out_channels)
        self.convShortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) if not self.equalInOut else None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class MaskBlock(nn.Module):
    """A basic resnet block with a mask to handle pruning"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, drop_rate=0.0):

        super(MaskBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.droprate = drop_rate
        self.equalInOut = (in_channels == out_channels)
        self.convShortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) if not self.equalInOut else None

        self.activation = Identity()
        self.activation.register_backward_hook(self._fisher)
        self.register_buffer('mask', None)

        self.input_shape = None
        self.output_shape = None
        self.flops = None
        self.params = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.got_shapes = False

        # Fisher method is called on backward passes
        self.running_fisher = 0

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        out = self.conv1(out if self.equalInOut else x)
        out = self.relu2(self.bn2(out))

        if self.mask is not None:
            out = out * self.mask[None, :, None, None]
        else:
            self._create_mask(x, out)

        out = self.activation(out)
        self.act = out  # activation of the intermediate channels, to be used by fisher pruning

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

    def _create_mask(self, x, out):
        self.mask = x.new_ones(out.shape[1])
        self.input_shape = x.size()
        self.output_shape = out.size()

    def _fisher(self, notused1, notused2, grad_output):
        act = self.act.detach()
        grad = grad_output[0].detach()

        g_nk = (act * grad).sum(-1).sum(-1)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        self.running_fisher += del_k

    def reset_fisher(self):
        self.running_fisher = 0 * self.running_fisher

    def cost(self):
        """computes the number of parameters of the model"""
        in_channels = self.in_channels
        out_channels = self.out_channels
        middle_channels = int(self.mask.sum().item())

        conv1_size = self.conv1.weight.size()
        conv2_size = self.conv2.weight.size()

        # convs
        self.params = in_channels * middle_channels * conv1_size[2] * conv1_size[3] + middle_channels * out_channels * \
                      conv2_size[2] * conv2_size[3]

        # batchnorms, assuming running stats are absorbed
        self.params += 2 * in_channels + 2 * middle_channels

        # skip
        if not self.equalInOut:
            self.params += in_channels * out_channels
        else:
            self.params += 0

    def compress_weights(self, verbose=True):
        """
        removes the pruned channels definitively rather than hiding them behind self.mask
        returns the number of channels left and the initial number of channels in the layer
        """
        middle_dim = int(self.mask.sum().item())
        full_middle_dim = self.mask.size(0)
        if verbose:
            print(f"{middle_dim} channels left out of {full_middle_dim}")

        if middle_dim is not 0:
            conv1 = nn.Conv2d(self.in_channels, middle_dim, kernel_size=3, stride=self.stride, padding=1, bias=False)
            conv1.weight = nn.Parameter(self.conv1.weight[self.mask == 1, :, :, :])

            # Batch norm 2 changes
            bn2 = nn.BatchNorm2d(middle_dim)
            bn2.weight = nn.Parameter(self.bn2.weight[self.mask == 1])
            bn2.bias = nn.Parameter(self.bn2.bias[self.mask == 1])
            bn2.running_mean = self.bn2.running_mean[self.mask == 1]
            bn2.running_var = self.bn2.running_var[self.mask == 1]

            conv2 = nn.Conv2d(middle_dim, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            conv2.weight = nn.Parameter(self.conv2.weight[:, self.mask == 1, :, :])
        else:
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


class NetworkBlock(nn.Module):
    """subblock (several resnet blocks with the same number of channels) of WideResNet module without bottlenecks"""
    def __init__(self, nb_layers, in_channels, out_channels, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_channels, out_channels, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_channels, out_channels, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(in_channels if i == 0 else out_channels, out_channels, stride if i == 0 else 1,
                                drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class NetworkBlockBottle(nn.Module):
    """subblock (several resnet blocks with the same number of channels) of WideResNet module with bottlenecks"""
    def __init__(self, nb_layers, in_channels, out_channels, mid_channels, block, stride, drop_rate=0.0):
        super(NetworkBlockBottle, self).__init__()
        self.layer = self._make_layer(block, in_channels, out_channels, mid_channels, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_channels, out_channels, mid_channels, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            if mid_channels[i] != 0:
                layers.append(
                    block(in_channels if i == 0 else out_channels, out_channels, mid_channels[i], stride if i == 0 else
                    1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """whole WideResNet-40-2 module without bottlenecks"""
    def __init__(self, depth, widen_factor, num_classes=10, drop_rate=0.0, mask=False):
        super(WideResNet, self).__init__()

        n_channels = [16, int(16 * widen_factor), int(32 * widen_factor), int(64 * widen_factor)]

        assert ((depth - 4) % 6 == 0)  # 4 = the initial conv layer + the 3 conv1*1 when we change the width
        n = (depth - 4) / 6

        if mask == 1:
            block = MaskBlock
        else:
            block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1, drop_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.nChannels = n_channels[3]

        # Count params that don't exist in blocks (conv1, bn1, fc)
        self.fixed_params = len(self.conv1.weight.view(-1)) + len(self.bn1.weight) + len(self.bn1.bias) + \
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
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avg_pool(out)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class WideResNetBottle(nn.Module):
    """whole WideResNet-40-2 module with bottlenecks"""
    def __init__(self, depth, widen_factor, num_classes=10, drop_rate=0.0, mid_channels=0.5):
        """
        the argument mid_channels might be a float scaling factor or a list containing the number of middle channels
        """
        super(WideResNetBottle, self).__init__()

        n_channels = [16, int(16 * widen_factor), int(32 * widen_factor), int(64 * widen_factor)]

        assert ((depth - 4) % 6 == 0)
        n = int((depth - 4) // 6)

        if isinstance(mid_channels, float):  # needs to be turned into a list
            mid_channels = [int(n_channels[1] * bottle_mult)]*n + [int(n_channels[2] * bottle_mult)]*n + \
                           [int(n_channels[3] * bottle_mult)]*n
        else:
            mid_channels = [int(i) for i in mid_channels]

        block = BottleBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlockBottle(n, n_channels[0], n_channels[1], mid_channels[:n], block, 1,
                                         drop_rate)
        # 2nd block
        self.block2 = NetworkBlockBottle(n, n_channels[1], n_channels[2], mid_channels[n:2*n], block, 2,
                                         drop_rate)
        # 3rd block
        self.block3 = NetworkBlockBottle(n, n_channels[2], n_channels[3], mid_channels[n:3*n], block, 2,
                                         drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.nChannels = n_channels[3]

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
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avg_pool(out)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
