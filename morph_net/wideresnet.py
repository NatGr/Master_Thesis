"""MorphNet on CIFAR-10 with a WideResNet architecture"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope


def wideresnet_block_concat(input_tensor, n_layers, n_channels_in, n_channels_out, stride, drop_rate, block_id):
    """function which returns a wideresnetblock """
    equal_in_out = n_channels_in == n_channels_out
    for i in range(int(n_layers)):
        if i == 0 and not equal_in_out:  # we need to have a relu before the skip connection because the skip conn is
            # a conv2d
            skip = layers.conv2d(input_tensor, n_channels_out, [1, 1], padding='same', stride=stride,
                                 scope="Skip_%d" % block_id)
        else:
            skip = input_tensor
        input_tensor = layers.conv2d(input_tensor, n_channels_out, [3, 3], padding='same', stride=stride,
                                     scope="Conv_%d_%d_1" % (block_id, i))
        if drop_rate > 0:
            input_tensor = layers.dropout(input_tensor, keep_prob=1.0 - drop_rate)
        input_tensor = layers.conv2d(input_tensor, n_channels_out, [3, 3], padding='same',
                                     scope="Conv_%d_%d_2" % (block_id, i))

        input_tensor = tf.add(skip, input_tensor)  # input of the next layer
        stride = 1  # stride is set to 1 for all the layers except the first

    return input_tensor


def wideresnet_concat(depth, widen_factor, input_tensor, num_classes=10, drop_rate=0.0):
    """function which builds a wideresnet (3*3->3*3) with the characteristics given as arguments"""
    n_channels = [16, int(16 * widen_factor), int(32 * widen_factor), int(64 * widen_factor)]

    assert ((depth - 4) % 6 == 0)
    n = (depth - 4) / 6

    params = {
        'trainable': True,
        'normalizer_fn': layers.batch_norm,
        'normalizer_params': {
            'scale': True
        }
    }

    with arg_scope([layers.conv2d], **params):
        conv1 = layers.conv2d(input_tensor, n_channels[0], [3, 3], padding='same', scope="Conv_0")

        block1 = wideresnet_block_concat(conv1, n, n_channels[0], n_channels[1], 1, drop_rate, 1)
        block2 = wideresnet_block_concat(block1, n, n_channels[1], n_channels[2], 2, drop_rate, 2)
        block3 = wideresnet_block_concat(block2, n, n_channels[2], n_channels[3], 2, drop_rate, 3)

        avg_pool = layers.avg_pool2d(block3, 8, stride=1)
        fc = layers.fully_connected(layers.flatten(avg_pool), num_classes, activation_fn=None)

    return fc


