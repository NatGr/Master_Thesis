"""
Wideresnet class to be pruned by NetAdapt
@Author: Nathan Greffe
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as fct
import numpy as np
import time
import os
import pickle


class WideResNet(nn.Module):
    """whole WideResNet module without bottlenecks"""

    def __init__(self, depth, widen_factor, device, prev_model=None, num_classes=10, drop_rate=0.0):
        """initializes the wrn"""
        super(WideResNet, self).__init__()

        self.n_channels = [16, int(16 * widen_factor), int(32 * widen_factor), int(64 * widen_factor)]  # channel sizes
        self.drop_rate = drop_rate
        self.device = device
        self.perf_table = None
        self.num_classes = num_classes
        self.total_cost = 0
        self.num_channels_dict = {}  # dict containing the number of channels remaining for each layer

        assert ((depth - 4) % 6 == 0)  # 4 = the initial conv layer + the 3 conv1*1 when we change the width
        self.n_blocks_per_subnet = int((depth - 4) / 6)

        self.compute_fisher_on = ["Conv_0"] + [f"Conv_{subnet_id}_{i}_{j}" for subnet_id in range(1, 4)
                                               for i in range(self.n_blocks_per_subnet)
                                               for j in range(1, 3)]  # no skip layers in this list since the activation
        # follows the add operation

        self.compute_table_on = ["Conv_0", "FC"] + [f"{pre}{i}{post}" for i in range(1, 4)
                                                    for pre, post in [("Skip_", ""), ("Conv_", "_0_1"),
                                                                      ("Conv_", "_0_2"), ("Conv_", "_1_1")]]

        self.to_prune = ["Conv_0"] + [f"Conv_{subnet_id}_{i}_1" for subnet_id in range(1, 4)
                                      for i in range(self.n_blocks_per_subnet)] + \
                        [f"Conv_{subnet_id}_0_2" for subnet_id in range(1, 4)]

        if prev_model is None:
            self._make_conv_layer("Conv_0", stride=1, n_channels_in=3, n_channels_out=self.n_channels[0])
            self.num_channels_dict["Conv_0"] = self.n_channels[0]
        else:
            self._make_conv_layer("Conv_0", stride=1, n_channels_in=3, n_channels_out=prev_model.Conv_0.out_channels)

        self._make_subnetwork(1, 1, prev_model)
        self._make_subnetwork(2, 2, prev_model)
        self._make_subnetwork(3, 2, prev_model)

        self.Avg_pool = nn.AvgPool2d(8)
        self.FC = nn.Linear(self.n_channels[3], self.num_classes) if prev_model is None else \
            nn.Linear(prev_model.FC.in_features, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _fisher(self, grad_output, act_name, running_fisher_name):
        """callback to compute the importance (on the loss) of a layer, act_name and running_fisher_name are the
         corresponding act and run_fish layer"""
        act = getattr(self, act_name).detach()
        grad = grad_output[0].detach()

        g_nk = (act * grad).sum(-1).sum(-1)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        running_fisher_prev = getattr(self, running_fisher_name)
        if running_fisher_prev is None:
            setattr(self, running_fisher_name, del_k)  # running average so as to get fisher_pruning on many different
            # data samples
        else:
            setattr(self, running_fisher_name, running_fisher_prev + del_k)

        # This is needed to avoid a memory leak caused by a circular reference between object and object.act_name
        setattr(self, act_name, None)

    def reset_fisher(self):
        for layer in self.compute_fisher_on:
            setattr(self, layer + "_run_fish", None)

    def _make_subnetwork(self, sub_net_id, stride, prev_model):
        """builds a wrn subnetwork
        :param sub_net_id: the id of the subnetwork, i.e. 1,2,...
        :param stride: the stride to apply
        :param prev_model: None or a previous instance of wideresnet we will use to determine the number of channels
        in each layer"""
        skip_name = f"Skip_{sub_net_id}"
        if prev_model is None:
            n_ch_in = self.n_channels[sub_net_id - 1]
            n_ch_out = self.n_channels[sub_net_id]
            setattr(self, skip_name, nn.Conv2d(n_ch_in, n_ch_out, kernel_size=3, stride=stride, padding=1, bias=False))
            self.num_channels_dict[skip_name] = n_ch_out
            for i in range(self.n_blocks_per_subnet):
                conv_1 = f"Conv_{sub_net_id}_{i}_1"
                conv_2 = f"Conv_{sub_net_id}_{i}_2"
                if i == 0:
                    self._make_conv_layer(conv_1, stride, n_ch_in, n_ch_out)
                else:
                    self._make_conv_layer(conv_1, 1, n_ch_out, n_ch_out)
                self.num_channels_dict[conv_1] = n_ch_out

                self._make_conv_layer(conv_2, 1, n_ch_out, n_ch_out)
                self.num_channels_dict[conv_2] = n_ch_out
        else:

            n_ch_in = getattr(prev_model, skip_name).in_channels
            n_ch_out = getattr(prev_model, skip_name).out_channels
            setattr(self, skip_name, nn.Conv2d(n_ch_in, n_ch_out, kernel_size=3, stride=stride, padding=1, bias=False))
            for i in range(self.n_blocks_per_subnet):
                for j in range(1, 3):
                    conv_name = f"Conv_{sub_net_id}_{i}_{j}"
                    conv_prev = getattr(prev_model, conv_name, None)
                    if conv_prev is None:
                        setattr(self, conv_name, None)
                    else:
                        self._make_conv_layer(conv_name, conv_prev.stride, conv_prev.in_channels,
                                              conv_prev.out_channels)

    def _make_conv_layer(self, name, stride, n_channels_in, n_channels_out):
        """initializes the attributes of a conv layer, creates a mask for the convolution iff new_mask is set to True"""
        setattr(self, name, nn.Conv2d(n_channels_in, n_channels_out, kernel_size=3, stride=stride, padding=1,
                                      bias=False))
        setattr(self, name + "_bn", nn.BatchNorm2d(n_channels_out))
        setattr(self, name + "_relu", nn.ReLU(inplace=True))
        setattr(self, name + "_run_fish", None)

    def _forward_subnet(self, subnet_id, x):
        out = x  # in case n_blocks_per_subnet == 0

        for i in range(self.n_blocks_per_subnet):
            name1 = f"Conv_{subnet_id}_{i}_1"
            name2 = f"Conv_{subnet_id}_{i}_2"
            conv_1 = getattr(self, name1, None)

            if conv_1 is not None:
                out = conv_1(x)
                out = getattr(self, name1 + "_relu")(getattr(self, name1 + "_bn")(out))

                setattr(self, name1 + "_act", out)
                out.register_hook(lambda x, act=name1 + "_act", rf=name1 + "_run_fish": self._fisher(x, act, rf))

                if self.drop_rate > 0:
                    out = fct.dropout(out, p=self.drop_rate, training=self.training)
                out = getattr(self, name2)(out)

                out = torch.add(out, x if i != 0 else getattr(self, f"Skip_{subnet_id}")(x))

                out = getattr(self, name2 + "_relu")(getattr(self, name2 + "_bn")(out))

                setattr(self, name2 + "_act", out)
                out.register_hook(lambda x, act=name2 + "_act", rf=name2 + "_run_fish": self._fisher(x, act, rf))
            else:
                out = x if i != 0 else getattr(self, f"Skip_{subnet_id}")(x)

            x = out  # because we will use it in the skip connection of next layer
        return out

    def forward(self, x):
        out = self.Conv_0(x)
        out = self.Conv_0_relu(self.Conv_0_bn(out))

        setattr(self, "Conv_0_act", out)
        out.register_hook(lambda x: self._fisher(x, "Conv_0_act", "Conv_0_run_fish"))

        for i in range(1, 4):
            out = self._forward_subnet(i, out)

        out = self.Avg_pool(out)
        out = out.view(out.size(0), -1)
        return self.FC(out)

    def load_table(self, name):
        """loads the pickle file coresponding to table and prints the total cost of the network"""
        with open(name, 'rb') as file:
            self.perf_table = pickle.load(file)

        self.total_cost = 0
        for layer_name in self.compute_table_on:
            if layer_name == "FC":
                self.total_cost += self.get_cost("FC", self.FC.in_features, 1)
            else:
                if layer_name.endswith("_0_2"):
                    factor = self.n_blocks_per_subnet  # number of times a similar conv exists in the network
                elif layer_name.endswith("_1_1"):
                    factor = self.n_blocks_per_subnet - 1
                else:
                    factor = 1
                layer = getattr(self, layer_name, None)
                if layer is not None:
                    if layer_name.startswith("Skip") or layer_name.endswith("_0_1"):
                        layer_name_table = f"Stride_{layer_name[5]}"
                    elif layer_name.endswith("_0_2") or layer_name.endswith("_1_1"):
                        layer_name_table = f"No_Stride_{layer_name[5]}"
                    else:
                        layer_name_table = layer_name

                    self.total_cost += factor * self.get_cost(layer_name_table, layer.in_channels, layer.out_channels)
                    # initially, the layers of the same type in the same subnetwork have the same number of channels

        print(f"the total cost of the model is: {self.total_cost} according to the perf table")

    def get_cost(self, layer_name, in_channel, out_channel):
        """ get the cost of layer layer_name when it has in_channel and out_channel channels, None means we return all
        input channels"""
        if in_channel is None:
            if out_channel is None:
                return self.perf_table[layer_name][:, :]
            else:
                return self.perf_table[layer_name][:, out_channel - 1]
        elif out_channel is None:
            return self.perf_table[layer_name][in_channel - 1, :]
        else:
            return self.perf_table[layer_name][in_channel - 1, out_channel - 1]

    def choose_which_channels(self, layer_name, num_channels, no_steps):
        """chooses which channels remains after the pruning
        :param layer_name: layer at which we remove filters
        :param num_channels: the number of channels to remove
        :param no_steps: the number of steps we make between each pruning
        :returns: the id of the channels that survives the pruning"""
        fisher = getattr(self, layer_name + "_run_fish")
        if layer_name.endswith("_0_2"):
            subnet = int(layer_name[5])
            for i in range(1, self.n_blocks_per_subnet):
                run_fish = getattr(self, f"Conv_{subnet}_{i}_2_run_fish")
                if run_fish is not None:  # we might have pruned this layer completely
                    fisher += run_fish
            num_layers = self.n_blocks_per_subnet
        else:
            num_layers = 1

        tot_loss = fisher.div(no_steps * num_layers)
        _, remaining_channels = torch.topk(tot_loss, k=tot_loss.size(0) - num_channels, largest=True, dim=0)

        remaining_channels, _ = torch.sort(remaining_channels)

        return remaining_channels

    def choose_num_channels(self, layer_name, cost_red_obj):
        """chooses how many channels to remove
        :param layer_name: layer at which we remove channels
        :param cost_red_obj: the given cost reduction objective (to attain or approach as much as possible)
        :returns: the number of channels to prune and achieved cost reduction or None, None if we cannot achieve
        cost_red_obj"""
        layer = getattr(self, layer_name, None)
        if layer is None:
            print(f"{layer_name} has been pruned away")
            return None, None
        init_nbr_out_channels = layer.out_channels
        print(f"{layer_name} has {init_nbr_out_channels} output channels")

        if layer_name == "Conv_0":  # when we remove one channel, we will influence Skip_1 and Conv_1_0_1 as well
            costs_array = self.get_cost(layer_name, 2, None)  # we will always have 3 input channels
            costs_array += self.get_cost("Stride_1", None, getattr(self, "Skip_1").out_channels)

            conv_1_0_1 = getattr(self, "Conv_1_0_1", None)
            if conv_1_0_1 is not None:  # the layer could have been entirely pruned away
                costs_array += self.get_cost("Stride_1", None, conv_1_0_1.out_channels)  # Conv_x_0_1 has same costs
                # as Skip_x

        elif layer_name.endswith("_0_2"):  # in that case, we have to prune several layers at the same time
            subnet = int(layer_name[5])

            # prev_layer, i.e. Skip_x
            costs_array = self.get_cost(f"Stride_{subnet}", getattr(self, f"Skip_{subnet}").in_channels, None)

            # next_layer
            if subnet != 3:
                skip_layer = f"Skip_{subnet + 1}"
                conv_next_subnet = f"Conv_{subnet + 1}_0_1"
                table_name = f"Stride_{subnet + 1}"
                costs_array += self.get_cost(table_name, None, getattr(self, skip_layer).out_channels)
                conv_y_0_1 = getattr(self, conv_next_subnet, None)
                if conv_y_0_1 is not None:
                    costs_array += self.get_cost(table_name, None, conv_y_0_1.out_channels)
            else:
                costs_array += self.get_cost("FC", None, 1)

            # subnet layers, we change the output of Conv_x_i_2 and the input of Conv_x_j_1 (j != 0)
            layer_name_table = f"No_Stride_{subnet}"
            for i in range(self.n_blocks_per_subnet):
                out_layer_name = f"Conv_{subnet}_{i}_2"
                out_layer = getattr(self, out_layer_name, None)
                if out_layer is not None:
                    costs_array += self.get_cost(layer_name_table, out_layer.in_channels, None)

            for j in range(1, self.n_blocks_per_subnet):
                in_layer_name = f"Conv_{subnet}_{j}_1"
                in_layer = getattr(self, in_layer_name, None)
                if in_layer is not None:
                    costs_array += self.get_cost(layer_name_table, None, in_layer.out_channels)

        else:  # layer_name = Conv_x_y_1, we only influence Conv_x_y_2 that is registered in perf_table as Conv_x_0_2
            # since we got here, Conv_x_y_1 cannot be none
            layer_name_table = f"No_Stride_{layer_name[5]}"
            next_layer = layer_name[:9] + "2"

            costs_array = self.get_cost(layer_name_table, getattr(self, layer_name).in_channels, None) + \
                self.get_cost(layer_name_table, None, getattr(self, next_layer).out_channels)

        # determines the number of filters
        prev_cost = costs_array[init_nbr_out_channels - 1]

        for rem_channels in range(init_nbr_out_channels - 1, 0, -1):  # could be made O(log(n)) instead
            cost_diff = prev_cost - costs_array[rem_channels - 1]  # -1 because we are directly accessing a cost_array
            if cost_diff > cost_red_obj:
                return init_nbr_out_channels - rem_channels, cost_diff

        if layer_name.startswith("Conv_") and layer_name.endswith("1") and prev_cost > cost_red_obj:
            return init_nbr_out_channels, prev_cost  # prune out the block

        print(f"{layer_name} cannot be pruned anymore")
        return None, None

    def prune_channels(self, layer_name, remaining_channels):
        """modifies the structure of self so as to remove the given channels at the given layer
        :param layer_name: the name of the layer that will be pruned
        :param remaining_channels: the offsets of the remaining channels"""

        if layer_name == "Conv_0":  # when we remove one channel, we will influence Skip_1 and Conv_1_0_1 as well
            self._remove_from_out_channels(layer_name, remaining_channels)
            if getattr(self, "Conv_1_0_1", None) is not None:
                self._remove_from_in_channels("Conv_1_0_1", remaining_channels)
            self._remove_from_in_channels("Skip_1", remaining_channels)

        elif layer_name.endswith("_0_2"):  # in that case, we have to prune several layers at the same time
            subnet = int(layer_name[5])

            # prev_layer, i.e. Skip_x
            self._remove_from_out_channels(f"Skip_{subnet}", remaining_channels)

            # next_layer
            if subnet != 3:
                conv_x_0_1 = f"Conv_{subnet + 1}_0_1"
                if getattr(self, conv_x_0_1, None) is not None:
                    self._remove_from_in_channels(conv_x_0_1, remaining_channels)
                self._remove_from_in_channels(f"Skip_{subnet + 1}", remaining_channels)
            else:
                self._remove_from_in_channels("FC", remaining_channels)

            # subnet layers
            for i in range(self.n_blocks_per_subnet):
                conv_x_i_2 = f"Conv_{subnet}_{i}_2"
                if getattr(self, conv_x_i_2, None) is not None:
                    self._remove_from_out_channels(conv_x_i_2, remaining_channels)

            for j in range(1, self.n_blocks_per_subnet):
                conv_x_j_1 = f"Conv_{subnet}_{j}_1"
                if getattr(self, conv_x_j_1, None) is not None:
                    self._remove_from_in_channels(conv_x_j_1, remaining_channels)

        else:  # name ends with conv_2_0_1, we only influence Conv_x_y_2
            next_layer_name = layer_name[:9] + "2"
            if remaining_channels.size(0) != 0:
                self._remove_from_out_channels(layer_name, remaining_channels)
                self._remove_from_in_channels(next_layer_name, remaining_channels)
            else:
                setattr(self, layer_name, None)
                setattr(self, next_layer_name, None)
                self.num_channels_dict[layer_name] = 0
                self.num_channels_dict[next_layer_name] = 0

    def load_bigger_state_dict(self, state_dict):
        """used to load a state dict that contains unused parameters (this is needed because when a layer is pruned away
         completely self.layer = None, it is not removed from state_dict, taken from
         https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113"""
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(pretrained_dict)

    def _remove_from_in_channels(self, layer_name, remaining_channels):
        """removes the input channels to the layer
        :param layer_name: the name of the layer that will be pruned
        :param remaining_channels: the offsets of the remaining channels"""
        layer = getattr(self, layer_name)
        num_remaining_channels = remaining_channels.size(0)
        if layer_name == "FC":
            new_layer = nn.Linear(num_remaining_channels, self.num_classes)
            new_layer.weight.data = layer.weight.data[:, remaining_channels]
            new_layer.bias.data = layer.bias.data

        else:
            new_layer = nn.Conv2d(in_channels=num_remaining_channels, out_channels=layer.out_channels,
                                  kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding,
                                  dilation=layer.dilation, groups=layer.groups, bias=layer.bias is not None)
            new_layer.weight.data = layer.weight.data[:, remaining_channels, :, :]  # out_ch * in_ch * height * width
            if new_layer.bias is not None:
                new_layer.bias.data = layer.bias.data
        setattr(self, layer_name, new_layer)

    def _remove_from_out_channels(self, layer_name, remaining_channels):
        """removes the output channels to the layer and adapts the corresponding batchnorm layers (in the case of
        Skip_x, Conv_x_0_2_bn is left unchanged)
        :param layer: the name of the layer that will be pruned
        :param remaining_channels: the offsets of the remaining channels"""
        layer = getattr(self, layer_name)
        num_remaining_channels = remaining_channels.size(0)
        self.num_channels_dict[layer_name] = num_remaining_channels

        # conv layer
        new_layer = nn.Conv2d(in_channels=layer.in_channels, out_channels=num_remaining_channels,
                              kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding,
                              dilation=layer.dilation, groups=layer.groups, bias=layer.bias is not None)

        new_layer.weight.data = layer.weight.data[remaining_channels, :, :, :]  # out_ch * in_ch * height * width
        if new_layer.bias is not None:
            new_layer.bias.data = layer.bias.data[remaining_channels]
        setattr(self, layer_name, new_layer)

        # batchnorm layer
        if layer_name[:4] != "Skip":
            bn_layer_name = layer_name + "_bn"
            bn_layer = getattr(self, bn_layer_name)
            new_bn_layer = nn.BatchNorm2d(num_remaining_channels, eps=bn_layer.eps, momentum=bn_layer.momentum,
                                          affine=bn_layer.affine, track_running_stats=bn_layer.track_running_stats)
            if new_bn_layer.affine:
                new_bn_layer.weight.data = bn_layer.weight.data[remaining_channels]
                new_bn_layer.bias.data = bn_layer.bias.data[remaining_channels]

            if new_bn_layer.track_running_stats:
                new_bn_layer.running_mean.data = bn_layer.running_mean.data[remaining_channels]
                new_bn_layer.running_var.data = bn_layer.running_var.data[remaining_channels]
                new_bn_layer.num_batches_tracked.data = bn_layer.num_batches_tracked.data
            setattr(self, bn_layer_name, new_bn_layer)
