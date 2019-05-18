"""mobilenetv1 model in tf.keras for 32*32 inputs"""
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from .commons import conv_2d_with_bn_relu, depthwise_conv_2d_with_bn_relu


def build_mobilenetv1(inputs, regularizer, blocks_per_subnet=(4, 4, 4), num_classes=10,
                      channels_per_subnet=(32, 64, 128)):
    """builds a mobilenetv1 model given a number of blocks per subnetwork"""
    x = conv_2d_with_bn_relu(16, kernel_size=3, regularizer=regularizer)(inputs)

    # like for WRN, only applies stride on first block of subnets 2 and 3:
    strides = [1, 2, 2]
    for i in range(3):
        x = mobilenetv1_block(x, channels_per_subnet[i], regularizer, strides[i])

        for _ in range(blocks_per_subnet[i] - 1):
            x = mobilenetv1_block(x, channels_per_subnet[i], regularizer)

    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    outputs = Dense(units=num_classes, activation='softmax', kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)(x)

    return Model(inputs=inputs, outputs=outputs)


def mobilenetv1_block(x_in, ch_out, regularizer, strides=1):
    """builds a mobilenetv1 block
    :param x_in: input of the block
    :param ch_out: number of output channels
    :param strides: the stride to apply in the first layer
    :param regularizer: the weight regularizer
    :return: the output of the block"""
    x = depthwise_conv_2d_with_bn_relu(strides, regularizer)(x_in)
    return conv_2d_with_bn_relu(ch_out, 1, regularizer)(x)

