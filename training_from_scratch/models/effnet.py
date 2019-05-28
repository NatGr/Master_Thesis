"""effnet model in tf.keras for 32*32 inputs"""
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, Dense, ReLU, LeakyReLU, Flatten, \
    Dropout, Add, DepthwiseConv2D
from tensorflow.keras.models import Model


def build_effnet(inputs, regularizer, blocks_per_subnet=(1, 1, 1), num_classes=10,
                 channels_per_subnet=(32, 64, 128, 256), expansion_rate=2):
    """builds a effnet model given a number of blocks per subnetwork"""
    x = Conv2D(channels_per_subnet[0], kernel_size=1, padding="same", use_bias=False)(inputs)
    x = BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(x)
    x = ReLU()(x)

    for i, nb_blocks in enumerate(blocks_per_subnet):
        ch_bottle = max(int(channels_per_subnet[i] * expansion_rate / 2), 6)
        ch_out = channels_per_subnet[i + 1]

        for _ in range(i-1):
            x = effnet_block(x, ch_bottle, ch_bottle, stride=False, regularizer=regularizer)
        x = effnet_block(x, ch_bottle, ch_out, stride=True, regularizer=regularizer)

    x = Flatten()(x)
    outputs = Dense(num_classes, activation=None, kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)
    return Model(inputs=inputs, outputs=outputs)


def effnet_block(x_in, ch_bottle, ch_out, stride, regularizer):
    """builds a effnet block
    :param x_in: input of the block
    :param ch_bottle: number of out channels of the first pointwise layer
    :param ch_out: number of output channels (if stride=False, ch_out is ignored and the number of output channels is
    ch_in
    :param stride: whether to have stride in this layer or not
    :param regularizer: the weight regularizer
    :return: the output of the block"""
    x = Conv2D(ch_bottle, kernel_size=1, padding="same", use_bias=False, kernel_regularizer=regularizer)(x_in)
    x = BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(x)
    x = LeakyReLU()(x)

    x = DepthwiseConv2D(kernel_size=(1, 3), padding="same", use_bias=False, kernel_regularizer=regularizer)(x)
    x = BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(x)
    x = ReLU()(x)

    if stride:
        x = MaxPool2D(pool_size=(1, 2), strides=(1, 2))(x)

    x = DepthwiseConv2D(kernel_size=(3, 1), padding="same", use_bias=False, kernel_regularizer=regularizer)(x)
    x = BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(x)
    x = ReLU()(x)

    if stride:
        x = Conv2D(ch_out, kernel_size=(2, 1), strides=(2, 1), padding="same", use_bias=False,
                   kernel_regularizer=regularizer)(x)
        x = BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(x)
        x = LeakyReLU()(x)

    return x

