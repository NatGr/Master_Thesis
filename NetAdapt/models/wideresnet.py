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
        self.perf_table = None
        self.num_classes = num_classes

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

        self._make_conv_layer("Conv_0", stride=1, n_channels_in=3, n_channels_out=self.n_channels[0])

        self._make_subnetwork(1, 1)
        self._make_subnetwork(2, 2)
        self._make_subnetwork(3, 2)

        self.Avg_pool = nn.AvgPool2d(8)
        self.FC = nn.Linear(self.n_channels[3], self.num_classes)

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
        setattr(self, running_fisher_name, running_fisher_prev + del_k)  # running average so as to get fisher_pruning
        # on many different data samples

    def reset_fisher(self):
        for layer in self.compute_fisher_on:
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
        setattr(self, name, nn.Conv2d(n_channels_in, n_channels_out, kernel_size=3, stride=stride, padding=1,
                                      bias=False))
        setattr(self, name + "_bn", nn.BatchNorm2d(n_channels_out))
        setattr(self, name + "_relu", nn.ReLU(inplace=True))
        if new_mask:
            self.register_buffer(name + '_mask', torch.ones(n_channels_out, device=self.device))
        setattr(self, name + "_activation", Identity())
        setattr(self, name + "_run_fish", 0)
        getattr(self, name + "_activation").register_backward_hook(lambda x, y, z: self._fisher(z, name + "_act",
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

        for i in range(1, 4):
            out = self._forward_subnet(i, out)

        out = self.Avg_pool(out)
        out = out.view(out.size(0), -1)
        return self.FC(out)

    def compute_table(self, file_name):
        """compute the associated real time inference table and save it as in a pickle file
        :param file_name: the name of the pickle file (without the extension), will be put in perf_tables folder"""

        # get for each element of the table the characteristics to test
        model_params = {}  # name -> (width/height, max_in_channels, max_out_channels, stride)

        for name in self.compute_table_on:
            if name != "FC":
                layer = getattr(self, name)
                if name == "Conv_0" or name == "Skip_1" or name == "Skip_2" or name.startswith("Conv_1") or \
                        name == "Conv_2_0_1":
                    width = 32
                elif name.startswith("Conv_2") or name == "Skip_3" or name == "Conv_3_0_1":
                    width = 16
                else:  # name.startswith("Conv_3") or name == "FC"
                    width = 8
                model_params[name] = (width, layer.in_channels, layer.out_channels, layer.stride[0])
            else:
                model_params["FC"] = (8, self.FC.in_features, 1, None)  # 1 and not self.FC.out_features because we
                # won't touch the final output layers

        # create the table
        perf_table = {}
        number_of_measures = 10

        self.make_conv_model(32, 32, 2)(torch.rand(1, 32, 32, 32, device=self.device))
        # so that pytorch caches whatever he needs

        for name in self.compute_table_on:
            (width, max_in_channels, max_out_channels, stride) = model_params[name]
            table_entry = np.zeros((max_in_channels, max_out_channels))

            for in_channels in range(1, max_in_channels + 1):
                for out_channels in range(1, max_out_channels + 1):
                    measures = np.zeros(number_of_measures)

                    for k in range(number_of_measures):
                        input_tensor = torch.rand(1, in_channels, width, width, device=self.device)
                        if name != "FC":
                            model = self.make_conv_model(in_channels, out_channels, stride)
                        else:
                            model = self.make_fc_model(in_channels, width)
                        begin = time.perf_counter()
                        model(input_tensor)
                        measures[k] = time.perf_counter() - begin

                    table_entry[in_channels - 1, out_channels - 1] = np.median(measures)
            perf_table[name] = table_entry

        with open(os.path.join('perf_tables', f"{file_name}.pickle"), 'wb') as file:
            pickle.dump(perf_table, file)

    def make_conv_model(self, in_channels, out_channels, stride):
        """creates a small sequential model composed of a convolution, a batchnorm and a relu activation
        the model is set to eval mode since it is used to measure evaluation time"""
        model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        model.to(self.device)
        model.eval()
        return model

    def make_fc_model(self, in_channels, width):
        """creates a small sequential model composed of an average pooling and a fully connected layer
        the model is set to eval mode since it is used to measure evaluation time"""

        class Flatten(nn.Module):  # not defined in pytorch
            def forward(self, input):
                return input.view(input.size(0), -1)

        model = nn.Sequential(
            nn.AvgPool2d(width),
            Flatten(),
            nn.Linear(in_channels, self.num_classes)
        )
        model.to(self.device)
        model.eval()
        return model

    def load_table(self, name):
        """loads the pickle file coresponding to table"""
        with open(os.path.join('perf_tables', f"{name}.pickle"), 'rb') as file:
            self.perf_table = pickle.load(file)

    def choose_which_filters(self, layer_name, cost_red_obj, prune_every):
        """chooses first how many and then which filters to remove
        :param layer_name: layer at which we remove filters
        :param cost_red_obj: the given cost reduction objective (to attain or approach as much as possible)
        :param prune_every: the number of steps we make between each pruning
        :returns: the new mask and the corresponding budget economised or (None, None) if we should not prune in
        that layer"""
        num_layers, inference_gains = self.choose_num_filters(layer_name, cost_red_obj)
        if num_layers is None:
            return None, None
        new_mask = getattr(self, layer_name + "_mask").clone()
        fisher = getattr(self, layer_name + "_run_fish")

        if layer_name.endswith("_0_2"):
            subnet = int(layer_name[5])
            for i in range(1, self.n_blocks_per_subnet):
                fisher += getattr(self, f"Conv_{subnet}_{i}_2_run_fish")
            num_layers = self.n_blocks_per_subnet
        else:
            num_layers = 1

        tot_loss = fisher.div(prune_every*num_layers) + 1e6 * (1 - new_mask)  # dummy value to get rid of already
        # pruned masks
        _, argmin = torch.topk(tot_loss, k=num_layers, largest=False, dim=0)
        new_mask[argmin] = 0

        return new_mask, inference_gains

    def choose_num_filters(self, layer_name, cost_red_obj):
        """chooses how many filters to remove at layer l to attain the given cost reduction objective or approach it
        as much as possible
        :returns: the number of layers we chose to remove and the corresponding inference time gains or (None, None) if
        we cannot prune in this layer anymore"""
        mask = getattr(self, layer_name + "_mask").detach()
        init_nbr_out_channels = mask.sum()

        if layer_name == "Conv_0":  # when we remove one channel, we will influence Skip_1 and Conv_1_0_1 as well
            conv_1_0_1_out_channels = getattr(self, "Conv_1_0_1_mask").detach().sum()
            skip_1_out_channels = getattr(self, "Conv_1_0_2_mask").detach().sum()

            costs_array = self.perf_table[layer_name][2, :]  # we will always have 3 input channels
            costs_array += self.perf_table["Skip_1"][:, skip_1_out_channels - 1] + \
                self.perf_table["Conv_1_0_1"][:, conv_1_0_1_out_channels - 1]

        elif layer_name.endswith("_0_2"):  # in that case, we have to prune several layers at the same time
            subnet = int(layer_name[5])

            # prev_layers, i.e. Skip_x and Conv_x_0_2
            skip_in_channel = getattr(self, f"Conv_{subnet - 1}_0_2_mask").detach().sum() if subnet != 1 else \
                getattr(self, "Conv_0_mask").detach().sum()
            conv_x_0_2_in_channels = getattr(self, f"Conv_{subnet}_0_1_mask").detach().sum()
            costs_array = self.perf_table[layer_name][conv_x_0_2_in_channels - 1, :] + \
                self.perf_table[f"Skip_{subnet}"][skip_in_channel - 1, :]

            # next_layer
            if subnet != 3:
                skip_layer = f"Skip_{subnet + 1}"
                conv_next_subnet = f"Conv_{subnet + 1}_0_1"
                skip_out_channel = getattr(self, skip_layer + "_mask").detach().sum()
                conv_xplus1_0_1_out_channels = getattr(self, conv_next_subnet + "_mask").detach().sum()
                costs_array += self.perf_table[skip_layer][:, skip_out_channel - 1] + \
                    self.perf_table[conv_next_subnet][:, conv_xplus1_0_1_out_channels - 1]
            else:
                costs_array += self.perf_table["FC"][:, 0]

            # subnet layers
                for i in range(1, self.n_blocks_per_subnet):
                    layer = f"Conv_{subnet}_{i}_1"
                    layer_out_channels = getattr(self, layer + "_mask").detach().sum()
                    costs_array += self.perf_table[layer][:, layer_out_channels]

        else:  # layer_name = Conv_x_y_1, we only influence Conv_x_y_2
            if layer_name[7] != "0":
                prev_layer = f"Conv_{layer_name[5]}_0_2"
            elif layer_name[5] == "1":
                prev_layer = "Conv_0"
            else:
                prev_layer = f"Conv_{int(layer_name[5]) - 1}_0_2"

            rem_in_channels = getattr(self, prev_layer + "_mask").detach().sum()

            next_layer = layer_name[:-1] + "2"
            next_layer_out_channels = getattr(self, next_layer + "_mask").detach().sum()

            costs_array = self.perf_table[layer_name][rem_in_channels - 1, :] + \
                self.perf_table[next_layer][:, next_layer_out_channels - 1]

        prev_cost = costs_array[init_nbr_out_channels - 1]

        for rem_channels in range(init_nbr_out_channels - 1, 0, -1):
            cost_diff = prev_cost - costs_array[rem_channels - 1]
            if cost_diff > cost_red_obj:
                return init_nbr_out_channels - rem_channels, cost_diff

        if layer_name.startswith("Conv_") and layer_name.endswith("1"):
            return init_nbr_out_channels, prev_cost  # prune out the block

        print("One of the channels has a surprisingly low number of layers")
        return None, None  # meaning we should not prune these layers because it would end with the network being cut
