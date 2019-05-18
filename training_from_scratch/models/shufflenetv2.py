"""shufflenetv2 model in tf.keras for 32*32 inputs"""
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, ReLU, Concatenate, Lambda
from tensorflow.keras.models import Model
from math import ceil
from .commons import depthwise_conv_2d_with_bn, conv_2d_with_bn_relu, channel_shuffle


def build_shufflenetv2(inputs, regularizer, blocks_per_subnet=(4, 4, 4), num_classes=10,
                       channels_per_subnet=(32, 64, 128)):
    """builds a shufflenetv2 model given a number of blocks per subnetwork"""
    x = conv_2d_with_bn_relu(16, kernel_size=3, regularizer=regularizer)(inputs)

    # like for WRN, only applies stride on first block of subnets 2 and 3:
    strides = [1, 2, 2]
    for i in range(3):
        x = shufflenetv2_block(x, channels_per_subnet[i], regularizer, strides[i])

        for _ in range(blocks_per_subnet[i] - 1):
            x = shufflenetv2_block(x, channels_per_subnet[i], regularizer)

    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    outputs = Dense(units=num_classes, activation='softmax', kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)(x)

    return Model(inputs=inputs, outputs=outputs)


def shufflenetv2_block(x_in, ch_out, regularizer, strides=1):
    """builds a shufflenetv2 block
    :param x_in: input of the block
    :param ch_out: number of output channels, ignored when strides != 1
    :param regularizer: the weight regularizer
    :param strides: the stride to apply in the first layer
    :return: the output of the block"""
    ch_in = x_in.shape.as_list()[3]

    if ch_in == ch_out and strides == 1:  # channel split
        ch_x = ceil(ch_in / 2)
        ch_out_x = ch_x
        x = Lambda(lambda z: z[:, :, :, :ch_x])(x_in)
        x_skip = Lambda(lambda z: z[:, :, :, ch_x:])(x_in)

    else:  # skip block
        ch_x = ch_in
        x = x_in

        if strides != 1:
            ch_out_x = ch_in
            ch_out_skip = ch_in
        else:
            ch_out_x = ceil(ch_out / 2)
            ch_out_skip = ch_out - ch_out_x

        x_skip = depthwise_conv_2d_with_bn(strides, regularizer)(x_in)
        x_skip = conv_2d_with_bn_relu(ch_out_skip, 1, regularizer)(x_skip)

    # main block
    x = conv_2d_with_bn_relu(ch_out_x // 2, 1, regularizer)(x)
    x = depthwise_conv_2d_with_bn(strides, regularizer)(x)
    x = conv_2d_with_bn_relu(ch_out_x, 1, regularizer)(x)

    # concatenate and channel shuffle
    x = Concatenate()([x, x_skip])
    return Lambda(channel_shuffle, arguments={'num_groups': 2})(x)  # 2 to mix the skip layer and the direct path
