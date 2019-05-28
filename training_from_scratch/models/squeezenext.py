"""squeezenext model in tf.keras for 32*32 inputs"""
from tensorflow.keras.layers import Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from .commons import conv_2d_with_bn_relu


def build_squeezenext(inputs, regularizer, blocks_per_subnet=(1, 1, 1), num_classes=10,
                      channels_per_subnet=(32, 64, 128)):
    """builds a squeezenet model given a number of blocks per subnetwork"""
    x = conv_2d_with_bn_relu(16, kernel_size=3, regularizer=regularizer, strides=1)(inputs)

    # like for WRN, only applies stride on first block of subnets 2 and 3:
    strides = [1, 2, 2]
    conv_3_1_first = True
    for i in range(3):
        x = squeezenext_block(x, channels_per_subnet[i], conv_3_1_first, regularizer, strides[i], skip_layer=True)
        conv_3_1_first = not conv_3_1_first

        for _ in range(blocks_per_subnet[i] - 1):
            x = squeezenext_block(x, channels_per_subnet[i], conv_3_1_first, regularizer)
            conv_3_1_first = not conv_3_1_first

    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    outputs = Dense(units=num_classes, activation=None, kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)(x)

    return Model(inputs=inputs, outputs=outputs)


def squeezenext_block(x_in, ch_out, conv_3_1_first, regularizer, strides=1, skip_layer=False):
    """builds a squeezenext block
    :param x_in: input of the block
    :param ch_out: number of output channels
    :param conv_3_1_first: wether the 3*1 or 1*3 conv comes first
    :param strides: the stride to apply in the first layer
    :param skip_layer: adds a skip layer (automatically done if strides > 1)
    :param regularizer: the weight regularizer
    :return: the output of the block"""
    x_skip = conv_2d_with_bn_relu(ch_out, kernel_size=1, strides=strides, regularizer=regularizer)(x_in) \
        if (strides > 1 or skip_layer) else x_in

    x = conv_2d_with_bn_relu(int(ch_out / 2), kernel_size=1, strides=strides, regularizer=regularizer)(x_in)
    x = conv_2d_with_bn_relu(int(ch_out / 4), kernel_size=1, regularizer=regularizer)(x)

    if conv_3_1_first:
        first_kernel = (3, 1)
        second_kernel = (1, 3)
    else:
        first_kernel = (1, 3)
        second_kernel = (3, 1)

    x = conv_2d_with_bn_relu(int(ch_out / 2), kernel_size=first_kernel, regularizer=regularizer)(x)
    x = conv_2d_with_bn_relu(int(ch_out / 2), kernel_size=second_kernel, regularizer=regularizer)(x)

    x = conv_2d_with_bn_relu(ch_out, kernel_size=1, regularizer=regularizer)(x)

    return Add()([x, x_skip])

