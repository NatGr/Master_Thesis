"""nasnet model in tf.keras for 32*32 inputs"""
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, ReLU, Concatenate, Lambda, Activation, \
    SeparableConv2D, BatchNormalization, Conv2D, ZeroPadding2D, Cropping2D, Add, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as keras_backend
from .commons import conv_2d_with_bn_relu


def build_nasnet(inputs, regularizer, blocks_per_subnet=(4, 4, 4), num_classes=10, channels_per_subnet=(32, 64, 128)):
    """builds a nasnet-A model given a number of blocks per subnetwork"""
    global _BN_DECAY, _BN_EPSILON, _REGULARIZER
    _BN_DECAY = 0.9
    _BN_EPSILON = 1e-5
    _REGULARIZER = regularizer

    x = conv_2d_with_bn_relu(16, kernel_size=3, regularizer=regularizer)(inputs)
    ip = inputs

    # like for WRN, only applies stride on first block of subnets 2 and 3:
    strides = [1, 2, 2]
    for i in range(3):
        if strides[i] != 1:  # this means that the real depth is one block smaller than declared
            x, ip = nasnet_red_block(x, ip, channels_per_subnet[i])

        for _ in range(blocks_per_subnet[i] - 1):
            x, ip = nasnet_normal_block(x, ip, channels_per_subnet[i])

    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    outputs = Dense(units=num_classes, activation='softmax', kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)(x)

    return Model(inputs=inputs, outputs=outputs)


"""rest of the file is a modified copy from https://github.com/titu1994/Keras-NASNet/blob/master/nasnet.py"""


def _separable_conv_block(ip, filters, kernel_size=(3, 3), strides=(1, 1)):
    """Adds 2 blocks of [relu-separable conv-batchnorm]
    # Arguments:
        ip: input tensor
        filters: number of output filters per layer
        kernel_size: kernel size of separable convolutions
        strides: strided convolution for downsampling
    # Returns:
        a Keras tensor
    """
    channel_dim = 1 if keras_backend.image_data_format() == 'channels_first' else -1

    x = Activation('relu')(ip)
    x = SeparableConv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False,
                        kernel_initializer='he_normal', kernel_regularizer=_REGULARIZER)(x)
    x = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON)(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters, kernel_size, padding='same', use_bias=False, kernel_initializer='he_normal',
                        kernel_regularizer=_REGULARIZER)(x)
    x = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON)(x)
    return x


def _adjust_block(p, ip, filters):
    """
    Adjusts the input `p` to match the shape of the `input`
    or situations where the output number of filters needs to
    be changed
    # Arguments:
        p: input tensor which needs to be modified
        ip: input tensor whose shape needs to be matched
        filters: number of output filters to be matched
    # Returns:
        an adjusted Keras tensor
    """
    channel_dim = 1 if keras_backend.image_data_format() == 'channels_first' else -1
    img_dim = 2 if keras_backend.image_data_format() == 'channels_first' else -2

    if p is None:
        p = ip

    elif p.shape.as_list()[img_dim] != ip.shape.as_list()[img_dim]:
        p = Activation('relu')(p)

        p1 = AveragePooling2D((1, 1), strides=(2, 2), padding='valid')(p)
        p1 = Conv2D(filters // 2, (1, 1), padding='same', use_bias=False, kernel_regularizer=_REGULARIZER,
                    kernel_initializer='he_normal')(p1)

        p2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
        p2 = Cropping2D(cropping=((1, 0), (1, 0)))(p2)
        p2 = AveragePooling2D((1, 1), strides=(2, 2), padding='valid')(p2)
        p2 = Conv2D(filters // 2, (1, 1), padding='same', use_bias=False, kernel_regularizer=_REGULARIZER,
                    kernel_initializer='he_normal')(p2)

        p = Concatenate(axis=channel_dim)([p1, p2])
        p = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON)(p)

    elif p.shape.as_list()[channel_dim] != filters:
        p = Activation('relu')(p)
        p = Conv2D(filters, (1, 1), strides=(1, 1), padding='same',
                   use_bias=False, kernel_regularizer=_REGULARIZER, kernel_initializer='he_normal')(p)
        p = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON)(p)
    return p


def nasnet_normal_block(ip, p, filters):
    """Adds a Normal cell for NASNet-A (Fig. 4 in the paper)
    # Arguments:
        ip: input tensor `x`
        p: input tensor `p`
        filters: number of output filters

    # Returns:
        the new x and ip
    """
    channel_dim = 1 if keras_backend.image_data_format() == 'channels_first' else -1

    p = _adjust_block(p, ip, filters)

    h = Activation('relu')(ip)
    h = Conv2D(filters, (1, 1), strides=(1, 1), padding='same',
               use_bias=False, kernel_initializer='he_normal', kernel_regularizer=_REGULARIZER)(h)
    h = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON)(h)

    x1_1 = _separable_conv_block(h, filters, kernel_size=(5, 5))
    x1_2 = _separable_conv_block(p, filters)
    x1 = Add()([x1_1, x1_2])

    x2_1 = _separable_conv_block(p, filters, (5, 5))
    x2_2 = _separable_conv_block(p, filters, (3, 3))
    x2 = Add()([x2_1, x2_2])

    x3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(h)
    x3 = Add()([x3, p])

    x4_1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(p)
    x4_2 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(p)
    x4 = Add()([x4_1, x4_2])

    x5 = _separable_conv_block(h, filters)
    x5 = Add()([x5, h])

    x = Concatenate(axis=channel_dim)([p, x1, x2, x3, x4, x5])
    return x, ip


def nasnet_red_block(ip, p, filters):
    """Adds a Reduction cell for NASNet-A (Fig. 4 in the paper)
    # Arguments:
        ip: input tensor `x`
        p: input tensor `p`
        filters: number of output filters
    # Returns:
        a Keras tensor
    """
    channel_dim = 1 if keras_backend.image_data_format() == 'channels_first' else -1

    p = _adjust_block(p, ip, filters)

    h = Activation('relu')(ip)
    h = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=_REGULARIZER)(h)
    h = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON)(h)

    x1_1 = _separable_conv_block(h, filters, (5, 5), strides=(2, 2))
    x1_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2))
    x1 = Add()([x1_1, x1_2])

    x2_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(h)
    x2_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2))
    x2 = Add()([x2_1, x2_2])

    x3_1 = AveragePooling2D((3, 3), strides=(2, 2), padding='same')(h)
    x3_2 = _separable_conv_block(p, filters, (5, 5), strides=(2, 2))
    x3 = Add()([x3_1, x3_2])

    x4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x1)
    x4 = Add()([x2, x4])

    x5_1 = _separable_conv_block(x1, filters, (3, 3))
    x5_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(h)
    x5 = Add()([x5_1, x5_2])

    x = Concatenate(axis=channel_dim)([x2, x3, x4, x5])
    return x, ip
