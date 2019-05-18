"""file containing functions used by several networks"""
from tensorflow.keras.layers import ReLU, BatchNormalization, Conv2D, AveragePooling2D, Flatten, Dense, \
    DepthwiseConv2D, Lambda, Concatenate
from tensorflow.keras import backend as keras_backend


def conv_2d_with_bn_relu(ch_out, kernel_size, regularizer, strides=1, relu_max_value=None):
    """laryer that encapsulates a conv2D followed by a batchnorm and a RELU"""
    return lambda x: ReLU(max_value=relu_max_value)(
        BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(
            Conv2D(ch_out, kernel_size=kernel_size, strides=strides, padding="same", use_bias=False,
                   kernel_regularizer=regularizer)(x)))


def depthwise_conv_2d_with_bn_relu(strides, regularizer, relu_max_value=None):
    """layer that encapsulates a depthwise conv2D followed by a batchnorm and a RELU"""
    return lambda x: ReLU(max_value=relu_max_value)(
        BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(
            DepthwiseConv2D(kernel_size=3, strides=strides, padding="same", use_bias=False,
                            kernel_regularizer=regularizer)(x)))


def depthwise_conv_2d_with_bn(strides, regularizer):
    """layer that encapsulates a depthwise conv2D followed by a batchnorm and without activations"""
    return lambda x: BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(
        DepthwiseConv2D(kernel_size=3, strides=strides, padding="same", use_bias=False,
                        kernel_regularizer=regularizer)(x))


def grouped_1_1_convolution_with_bn(x, ch_out, num_groups, regularizer, use_relu):
    """layer that encapsulates a grouped conv2D with kernel_size 1*1 followed by a batchnorm and
    a ReLU activation if use_relu is True.
    This layer does not follow the same API as the others (takes the input as argument) for simplicity since we have
    to code grouped convolutions ourselves
    parts of this code were adapted from https://github.com/scheckmedia/keras-shufflenet/blob/master/shufflenet.py
    Our version can handle the case where ch_in % num_groups != 0, this is not the case for ch_out"""
    if num_groups == 1:
        grouped_conv = Conv2D(ch_out, kernel_size=1, padding="same", use_bias=False, kernel_regularizer=regularizer)(x)
    else:
        ch_in = x.shape.as_list()[3]
        num_ch_in_group = int(ch_in // num_groups)
        num_ch_out_group = int(ch_out // num_groups)
        groups = []

        assert ch_out % num_groups == 0

        for i in range(num_groups):
            offset = i * num_ch_in_group

            if i == num_groups - 1:  # the last group might be a bit wider if the divisions do not result in integers
                num_ch_in_group += ch_in % num_groups

            group = Lambda(lambda z: z[:, :, :, offset: offset + num_ch_in_group])(x)
            groups.append(Conv2D(num_ch_out_group, kernel_size=1, padding="same", use_bias=False,
                                 kernel_regularizer=regularizer)(group))
        grouped_conv = Concatenate()(groups)

    bn_layer = BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(grouped_conv)
    if use_relu:
        return ReLU()(bn_layer)
    else:
        return bn_layer


def channel_shuffle(x, num_groups):
    """
    shuffles a channel
    code taken from https://github.com/scheckmedia/keras-shufflenet/blob/master/shufflenet.py
    """
    height, width, in_channels = x.shape.as_list()[1:]
    channels_per_group = in_channels // num_groups

    x = keras_backend.reshape(x, [-1, height, width, num_groups, channels_per_group])
    x = keras_backend.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
    x = keras_backend.reshape(x, [-1, height, width, in_channels])

    return x
