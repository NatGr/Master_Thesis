"""MorphNet on CIFAR-10 with a WideResNet architecture
Author: Nathan Greffe"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pickle
import os

layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope


def number_channels_from_pickle(filename):
    """function returning the number of chanels that were not pruned inside of a pickle file
    it also prints the number of channels left per layer"""
    file = open(os.path.join('morph_net', 'pickle', filename), 'rb')
    dict_channels = pickle.load(file)
    num_channels = 0
    for name, channels_ratio in dict_channels.items():
        print(name, ": %d/%d" % channels_ratio)
        num_channels += channels_ratio[0]
    return num_channels


def wideresnet_block_concat(input_tensor, n_layers, n_channels_out, stride, drop_rate, block_id, channels_dict):
    """
    function which returns a wideresnetblock
    if channels_dict is not None, it is used to compute the number of output channels for each layer instead of the
    n_channels_out arguments, n_channels_out should still be correct when using channels_dict
    """
    no_channel_dict = channels_dict is None

    for i in range(int(n_layers)):
        conv1_name = "Conv_%d_%d_1" % (block_id, i)
        if i == 0:  # we do not use equal_in_out here since it introduces complications with channels_dict and does
            # not change anything for wideresnets of widen_factor != 1
            skip = layers.conv2d(input_tensor, n_channels_out, [1, 1], padding='same', stride=stride,
                                 scope="Skip_%d" % block_id)
        else:
            skip = input_tensor

        if not no_channel_dict and (conv1_name not in channels_dict or channels_dict[conv1_name] == 0):  # if the layer
            # was entirely pruned away, we should skip it in the network
            input_tensor = skip  # input of the next layer
            stride = 1  # stride is set to 1 for all the layers except the first
        else:
            input_tensor = layers.conv2d(input_tensor, n_channels_out if no_channel_dict else channels_dict[conv1_name],
                                         [3, 3], padding='same', stride=stride, scope=conv1_name)
            if drop_rate > 0:
                input_tensor = layers.dropout(input_tensor, keep_prob=1.0 - drop_rate)
            input_tensor = layers.conv2d(input_tensor, n_channels_out, [3, 3], padding='same',
                                         scope="Conv_%d_%d_2" % (block_id, i))

            input_tensor = tf.add(skip, input_tensor)  # input of the next layer
            stride = 1  # stride is set to 1 for all the layers except the first

    return input_tensor


def wideresnet_concat(depth, widen_factor, input_tensor, num_classes=10, drop_rate=0.0, channels_dict=None):
    """
    function which builds a wideresnet (3*3->3*3) with the characteristics given as arguments
    if channels_dict is not None, it is used to compute the number of output channels for each layer instead of the
    default number from the wideresnet paper"""
    assert ((depth - 4) % 6 == 0)
    n = (depth - 4) / 6

    first_layer_name = "Conv_0"

    if channels_dict is None:
        n_channels = [16, int(16 * widen_factor), int(32 * widen_factor), int(64 * widen_factor)]
    else:
        n_channels = [channels_dict[first_layer_name], channels_dict["Skip_1"],
                      channels_dict["Skip_2"], channels_dict["Skip_3"]]

    params = {
        'trainable': True,
        'normalizer_fn': layers.batch_norm,
        'normalizer_params': {
            'scale': True
        }
    }

    with arg_scope([layers.conv2d], **params):
        conv1 = layers.conv2d(input_tensor, n_channels[0], [3, 3], padding='same', scope=first_layer_name)

        block1 = wideresnet_block_concat(conv1, n, n_channels[1], 1, drop_rate, 1, channels_dict)
        block2 = wideresnet_block_concat(block1, n, n_channels[2], 2, drop_rate, 2, channels_dict)
        block3 = wideresnet_block_concat(block2, n, n_channels[3], 2, drop_rate, 3, channels_dict)

        avg_pool = layers.avg_pool2d(block3, 8, stride=1)
        fc = layers.fully_connected(layers.flatten(avg_pool), num_classes, activation_fn=None)

    return fc


