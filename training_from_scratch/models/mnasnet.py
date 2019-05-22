"""mnasnet model in tf.keras for 32*32 inputs"""
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Add, BatchNormalization, Conv2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
from .commons import conv_2d_with_bn_relu
from .mobilenetv2 import mobilenetv2_block


def build_mnasnet(inputs, regularizer, blocks_per_subnet=(4, 4, 4), num_classes=10,
                  channels_per_subnet=(16, 32, 64, 128), expansion_factor=4, use_dropout=False, se_factor=0):
    """builds a mnasnet network for cifar-10, we hypothesizes that the first block of mnasnet was as is it only because
    of input feature map resolution, thus this is basically a mobilenetv2 with 5*5 convolutions in the middle. We also
    use expansion_factor/2 in the first subnetwork"""
    x = conv_2d_with_bn_relu(16, kernel_size=3, regularizer=regularizer, relu_max_value=6)(inputs)

    # like for WRN, only applies stride on first block of subnets 2 and 3:
    strides = [1, 2, 2]
    for i in range(3):
        depthwise_kernel_size = [3] * blocks_per_subnet[i] if i == 0 else ([5] * (blocks_per_subnet[i] - 1) + [3])
        num_channels = np.linspace(channels_per_subnet[i], channels_per_subnet[i+1],
                                   blocks_per_subnet[i]+1).astype(np.int)
        x = mobilenetv2_block(x, num_channels[0], num_channels[1],
                              expansion_factor // 2 if i == 0 else expansion_factor, regularizer, strides[i],
                              depthwise_kernel_size[0], se_factor=se_factor)

        for j in range(1, blocks_per_subnet[i]):
            x = mobilenetv2_block(x, num_channels[j], num_channels[j+1], expansion_factor, regularizer,
                                  mid_conv_size=depthwise_kernel_size[j], se_factor=se_factor)

    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    if use_dropout:
        x = Dropout(rate=.2)(x)
    outputs = Dense(units=num_classes, activation='softmax', kernel_regularizer=regularizer,
                    bias_regularizer=regularizer)(x)

    return Model(inputs=inputs, outputs=outputs)
