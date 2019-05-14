"""mobilenetv1 model in tf.keras for 32*32 inputs"""
from tensorflow.keras.layers import ReLU, BatchNormalization, Conv2D, AveragePooling2D, Flatten, Dense, \
    DepthwiseConv2D
from tensorflow.keras.models import Model


def build_mobilenetv1(inputs, blocks_per_subnet=(4, 4, 4), num_classes=10, channels_per_subnet=(32, 64, 128)):
    """builds a mobilenetv1 model given a number of blocks per subnetwork"""
    x = conv_2d_with_bn_relu(16, kernel_size=3)(inputs)

    # like for WRN, only applies stride on first block of subnets 2 and 3:
    strides = [1, 2, 2]
    for i in range(3):
        x = mobilenetv1_block(x, channels_per_subnet[i], strides[i])

        for _ in range(blocks_per_subnet[i] - 1):
            x = mobilenetv1_block(x, channels_per_subnet[i])

    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    outputs = Dense(units=num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)


def mobilenetv1_block(x_in, ch_out, strides=1):
    """builds a mobilenetv1 block
    :param x_in: input of the block
    :param ch_out: number of output channels
    :param strides: the stride to apply in the first layer
    :return: the output of the block"""
    x = depthwise_conv_2d_with_bn_relu(strides)(x_in)
    return conv_2d_with_bn_relu(ch_out, kernel_size=1)(x)


def conv_2d_with_bn_relu(ch_out, kernel_size):
    """laryer that encapsulated a conv2D followed by a batchnorm and a RELU"""
    return lambda x: ReLU()(BatchNormalization()(Conv2D(ch_out, kernel_size=kernel_size, strides=1, padding="same",
                                                        use_bias=False)(x)))


def depthwise_conv_2d_with_bn_relu(strides):
    """laryer that encapsulated a conv2D followed by a batchnorm and a RELU"""
    return lambda x: ReLU()(BatchNormalization()(DepthwiseConv2D(kernel_size=3, strides=strides, padding="same",
                                                 use_bias=False)(x)))

