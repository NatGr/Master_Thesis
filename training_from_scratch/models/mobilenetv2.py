"""mobilenetv2 model in tf.keras for 32*32 inputs"""
from tensorflow.keras.layers import ReLU, BatchNormalization, Conv2D, AveragePooling2D, Flatten, Dense, \
    DepthwiseConv2D, Add
from tensorflow.keras.models import Model
import numpy as np


def build_mobilenetv2(inputs, regularizer, blocks_per_subnet=(4, 4, 4), num_classes=10,
                      channels_per_subnet=(16, 32, 64, 128), expansion_factor=4):
    """builds a mobilenetv1 model given a number of blocks per subnetwork, like for the imagenet version, we will still
    have an expansion factor of 1 for the first layer and we will have a smooth progression into the number of channels
    per block"""
    x = conv_2d_with_bn_relu(16, kernel_size=3, regularizer=regularizer)(inputs)

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
    outputs = Dense(units=num_classes, activation='softmax', kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)(x)

    return Model(inputs=inputs, outputs=outputs)


def mobilenetv2_block(x_in, ch_in, ch_out, expansion_factor, regularizer, strides=1):
    """builds a mobilenetv1 block
    :param x_in: input of the block
    :param ch_in: number of input channels
    :param ch_out: number of output channels
    :param expansion_factor: factor by which the inner channels are multiplied
    :param regularizer: the weight regularizer
    :param strides: the stride to apply in the first layer
    :return: the output of the block"""
    x = conv_2d_with_bn_relu(ch_out=expansion_factor*ch_in, kernel_size=1, regularizer=regularizer)(x_in)
    x = depthwise_conv_2d_with_bn_relu(strides, regularizer=regularizer)(x)
    x = BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(
        Conv2D(ch_out, kernel_size=1, strides=1, padding="same", use_bias=False, kernel_regularizer=regularizer)(x))
    return x if strides == 2 or ch_in != ch_out else Add()([x, x_in])


def conv_2d_with_bn_relu(ch_out, kernel_size, regularizer):
    """laryer that encapsulated a conv2D followed by a batchnorm and a RELU"""
    return lambda x: ReLU(max_value=6)(BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(
        Conv2D(ch_out, kernel_size=kernel_size, strides=1, padding="same", use_bias=False,
               kernel_regularizer=regularizer)(x)))


def depthwise_conv_2d_with_bn_relu(strides, regularizer):
    """laryer that encapsulated a conv2D followed by a batchnorm and a RELU"""
    return lambda x: ReLU(max_value=6)(BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(
        DepthwiseConv2D(kernel_size=3, strides=strides, padding="same", use_bias=False,
                        kernel_regularizer=regularizer)(x)))

