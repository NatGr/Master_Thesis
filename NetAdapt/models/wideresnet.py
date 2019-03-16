"""
Wideresnet class to be pruned by NetAdapt
@Author: Nathan Greffe
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as fct


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class WideResNet(nn.Module):
    """whole WideResNet module without bottlenecks"""
    def __init__(self, depth, widen_factor, device, num_classes=10, drop_rate=0.0):
        """initializes the wrn"""
        super(WideResNet, self).__init__()

        self.n_channels = [16, int(16 * widen_factor), int(32 * widen_factor), int(64 * widen_factor)]  # channel sizes
        self.drop_rate = drop_rate
        self.device = device

        assert ((depth - 4) % 6 == 0)  # 4 = the initial conv layer + the 3 conv1*1 when we change the width
        self.n_blocks_per_subnet = (depth - 4) / 6

        self.layers_idx = ["Conv_0"] + [f"Conv_{subnet_id}_{i}_{j}" for subnet_id in range(1,4) for i in range(self.n_blocks_per_subnet)
                                   for j in range(1,3)]  # do not contain the skip_conv_layers

        self._make_conv_layer("Conv_0", stride=1, n_channels_in=3, n_channels_out=self.n_channels[0])

        self._make_subnetwork(1, 1)
        self._make_subnetwork(2, 2)
        self._make_subnetwork(3, 2)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.n_channels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _fisher(self, notused1, notused2, grad_output, act_name, running_fisher_name):
        """callback to compute the importance (on the loss) of a layer, act_name and running_fisher_name are the
         corresponding act and run_fish layer"""
        act = getattr(self, act_name).detach()
        grad = grad_output[0].detach()

        g_nk = (act * grad).sum(-1).sum(-1)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        running_fisher_prev = getattr(self, running_fisher_name)
        setattr(self, running_fisher_name, running_fisher_prev + del_k)

    def reset_fisher(self):
        for layer in self.layers_idx:
            setattr(self, layer, 0)

    def _make_subnetwork(self, sub_net_id, stride):
        """builds a wrn subnetwork
        :param sub_net_id: the id of the subnetwork, i.e. 1,2,...
        :param stride: the stride to apply"""
        n_ch_in = self.n_channels[sub_net_id - 1]
        n_ch_out = self.n_channels[sub_net_id]
        setattr(self, f"Skip_{sub_net_id}",
                nn.Conv2d(n_ch_in, n_ch_out, kernel_size=3, stride=stride, padding=1, bias=False))  # skip conv layer
        for i in range(self.n_blocks_per_subnet):
            if i == 0:
                self._make_conv_layer(f"Conv_{sub_net_id}_{i}_1", stride, n_ch_in, n_ch_out)
            else:
                self._make_conv_layer(f"Conv_{sub_net_id}_{i}_1", 1, n_ch_out, n_ch_out)
            self._make_conv_layer(f"Conv_{sub_net_id}_{i}_2", 1, n_ch_out, n_ch_out, new_mask=False)

    def _make_conv_layer(self, name, stride, n_channels_in, n_channels_out, new_mask=True):
        """initializes the attributes of a conv layer, creates a mask for the convolution iff new_mask is set to True"""
        setattr(self, name, nn.Conv2d(n_channels_in, n_channels_out, kernel_size=3, stride=stride, padding=1, bias=False))
        setattr(self, name + "_bn", nn.BatchNorm2d(self.n_channels_out))
        setattr(self, name + "_relu", nn.ReLU(inplace=True))
        if new_mask:
            setattr(self, name + "_mask", torch.ones(n_channels_out, device=self.device))
        setattr(self, name + "_activation", Identity())
        setattr(self, name + "_run_fish", 0)
        getattr(self, name + "_activation").register_backward_hook(lambda x, y, z: self._fisher(x, y, z, name + "_act",
                                                                                   name + "_run_fish"))

    def _forward_subnet(self, subnet_id, x):
        name_shared_mask = f"Conv_{subnet_id}_0_2"  # name of the mask that is shared by half of the layers
        out = x
        
        for i in range(self.n_blocks_per_subnet):
            name_1 = f"Conv_{subnet_id}_{i}_1"
            name_2 = f"Conv_{subnet_id}_{i}_2"

            out = getattr(self, name_1)(x)
            out = getattr(self, name_1 + "_relu")(getattr(self, name_1 + "_bn")(out))
            out = out * getattr(self, name_1 + "_mask")[None, :, None, None]
            out = getattr(self, name_1 + "_activation")(out)
            setattr(self, name_1 + "_act", out)
            if self.droprate > 0:
                out = fct.dropout(out, p=self.droprate, training=self.training)
            out = getattr(self, name_2)(out)

            out = torch.add(out, x if i == 0 else getattr(self, f"Skip_{subnet_id}")(x))

            out = getattr(self, name_2 + "_relu")(getattr(self, name_2 + "_bn")(out))
            out = out * getattr(self, name_shared_mask)[None, :, None, None]
            out = getattr(self, name_2 + "_activation")(out)
            setattr(self, name_2 + "_act", out)

            x = out  # because we will use it in the skip connection of next layer
        return out

    def forward(self, x):
        out = self.Conv_0(x)
        out = self.Conv_0_relu(self.Conv_0_bn(out))
        out = out * self.Conv_0_mask[None, :, None, None]
        out = self.Conv_0_activation(out)
        self.Conv_0_act = out

        for i in range(1,4):
            out = self._forward_subnet(i, out)

        out = self.avg_pool(out)
        out = out.view(-1, self.n_channels[3])
        return self.fc(out)

    def get_table(self):
        pass

