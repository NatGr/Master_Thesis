"""shufflenetv1 model in tf.keras for 32*32 inputs"""
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, ReLU, Add, Concatenate, Lambda
from tensorflow.keras.models import Model
from .commons import grouped_1_1_convolution_with_bn, depthwise_conv_2d_with_bn, conv_2d_with_bn_relu, channel_shuffle


def build_shufflenetv1(inputs, regularizer, blocks_per_subnet=(4, 4, 4), num_classes=10, num_groups=3,
                       channels_per_subnet=(32, 64, 128)):
    """builds a shufflenetv1 model given a number of blocks per subnetwork"""
    x = conv_2d_with_bn_relu(24, kernel_size=3, regularizer=regularizer)(inputs)  # 24 so as to have something that
    # works better with different num_groups

    # like for WRN, only applies stride on first block of subnets 2 and 3:
    strides = [1, 2, 2]
    for i in range(3):
        x = shufflenetv1_block(x, channels_per_subnet[i], num_groups, regularizer, strides[i])

        for _ in range(blocks_per_subnet[i] - 1):
            x = shufflenetv1_block(x, channels_per_subnet[i], num_groups, regularizer)

    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    outputs = Dense(units=num_classes, activation='softmax', kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)(x)

    return Model(inputs=inputs, outputs=outputs)


def shufflenetv1_block(x_in, ch_out, num_groups, regularizer, strides=1):
    """builds a shufflenetv1 block
    :param x_in: input of the block
    :param ch_out: number of output channels, ignored when strides != 1
    :param num_groups: number of groups in the 1*1 convolutions
    :param regularizer: the weight regularizer
    :param strides: the stride to apply in the first layer
    :return: the output of the block"""
    ch_in = x_in.shape.as_list()[3]

    if strides != 1:
        ch_out = ch_in

    # main branch
    x = grouped_1_1_convolution_with_bn(x_in, int(ch_out/4), num_groups, regularizer, True)
    x = Lambda(channel_shuffle, arguments={'num_groups': num_groups})(x)
    x = depthwise_conv_2d_with_bn(strides, regularizer)(x)
    x = grouped_1_1_convolution_with_bn(x, ch_out, num_groups, regularizer, False)

    # skip_connection
    if strides == 2:
        x = Concatenate()([x, AveragePooling2D(strides=2, pool_size=3, padding="same")(x_in)])
    elif ch_in == ch_out:  # if ch_in != ch_out, we don't put a skip_connection
        x = Add()([x, x_in])

    return ReLU()(x)
