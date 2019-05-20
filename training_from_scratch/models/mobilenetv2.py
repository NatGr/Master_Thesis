"""mobilenetv2 model in tf.keras for 32*32 inputs"""
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Add, BatchNormalization, Conv2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
from .commons import conv_2d_with_bn_relu, depthwise_conv_2d_with_bn_relu


def build_mobilenetv2(inputs, regularizer, blocks_per_subnet=(4, 4, 4), num_classes=10,
                      channels_per_subnet=(16, 32, 64, 128), expansion_factor=4, use_dropout=False):
    """builds a mobilenetv2 model given a number of blocks per subnetwork, like for the imagenet version, we will still
    have an expansion factor of 1 for the first layer and we will have a smooth progression into the number of channels
    per block"""
    x = conv_2d_with_bn_relu(16, kernel_size=3, regularizer=regularizer, relu_max_value=6)(inputs)

    # like for WRN, only applies stride on first block of subnets 2 and 3:
    strides = [1, 2, 2]
    for i in range(3):
        num_channels = np.linspace(channels_per_subnet[i], channels_per_subnet[i+1],
                                   blocks_per_subnet[i]+1).astype(np.int)
        x = mobilenetv2_block(x, num_channels[0], num_channels[1],
                              1 if i == 0 else expansion_factor, regularizer, strides[i])

        for j in range(1, blocks_per_subnet[i]):
            x = mobilenetv2_block(x, num_channels[j], num_channels[j+1], expansion_factor, regularizer)

    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    if use_dropout:
        x = Dropout(rate=.2)(x)
    outputs = Dense(units=num_classes, activation='softmax', kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)(x)

    return Model(inputs=inputs, outputs=outputs)


def mobilenetv2_block(x_in, ch_in, ch_out, expansion_factor, regularizer, strides=1, mid_conv_size=3):
    """builds a mobilenetv1 block
    :param x_in: input of the block
    :param ch_in: number of input channels
    :param ch_out: number of output channels
    :param expansion_factor: factor by which the inner channels are multiplied
    :param regularizer: the weight regularizer
    :param strides: the stride to apply in the first layer
    :param mid_conv_size: the size of the kernel of the depthwise separable convolution layer in the middle of the block
    :return: the output of the block"""
    x = conv_2d_with_bn_relu(ch_out=expansion_factor*ch_in, kernel_size=1, regularizer=regularizer,
                             relu_max_value=6)(x_in)

    x = depthwise_conv_2d_with_bn_relu(strides, regularizer=regularizer, relu_max_value=6, kernel_size=mid_conv_size)(x)

    x = BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(
        Conv2D(ch_out, kernel_size=1, strides=1, padding="same", use_bias=False, kernel_regularizer=regularizer)(x))

    return x if strides == 2 or ch_in != ch_out else Add()([x, x_in])
