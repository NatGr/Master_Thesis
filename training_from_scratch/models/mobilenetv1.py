"""mobilenetv1 model in tf.keras for 32*32 inputs"""
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Add, Dropout
from tensorflow.keras.models import Model
from .commons import conv_2d_with_bn_relu, depthwise_conv_2d_with_bn_relu, se_block


def build_mobilenetv1(inputs, regularizer, blocks_per_subnet=(4, 4, 4), num_classes=10,
                      channels_per_subnet=(32, 64, 128), use_5_5_filters=False, use_dropout=False,
                      se_factor=0, add_skip=False):
    """builds a mobilenetv1 model given a number of blocks per subnetwork"""
    x = conv_2d_with_bn_relu(16, kernel_size=3, regularizer=regularizer)(inputs)

    # like for WRN, only applies stride on first block of subnets 2 and 3:
    strides = [1, 2, 2]
    for i in range(3):
        depthwise_kernel_size = [3] * blocks_per_subnet[i] if i == 0 or not use_5_5_filters \
            else ([5] * (blocks_per_subnet[i] - 1) + [3])
        x = mobilenetv1_block(x, channels_per_subnet[i], regularizer, strides[i], kern_size=depthwise_kernel_size[0],
                              se_factor=se_factor, add_skip=add_skip)

        for j in range(1, blocks_per_subnet[i]):
            x = mobilenetv1_block(x, channels_per_subnet[i], regularizer, kern_size=depthwise_kernel_size[j],
                                  se_factor=se_factor, add_skip=add_skip)

    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    if use_dropout:
        x = Dropout(rate=.2)(x)
    outputs = Dense(units=num_classes, activation='softmax', kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)(x)

    return Model(inputs=inputs, outputs=outputs)


def mobilenetv1_block(x_in, ch_out, regularizer, strides=1, kern_size=3, se_factor=0, add_skip=False):
    """builds a mobilenetv1 block
    :param x_in: input of the block
    :param ch_out: number of output channels
    :param strides: the stride to apply in the first layer
    :param regularizer: the weight regularizer
    :param kern_size: the size of the Dwise convolution kernel
    :param se_factor: the reduction factor to apply to the se block
    (no se block created if se_factor == 0 or strides != 1)
    :param add_skip: adds a skip layer to the block (ignored if strides != 1 or ch_in != ch_out)
    :return: the output of the block"""
    x = depthwise_conv_2d_with_bn_relu(strides, regularizer, kernel_size=kern_size)(x_in)
    x = conv_2d_with_bn_relu(ch_out, 1, regularizer)(x)

    if strides == 1:
        if se_factor != 0:
            x = se_block(x, ch_out, se_factor, regularizer)

        if add_skip and x_in.shape.as_list()[3] == ch_out:
            x = Add()([x_in, x])

    return x
