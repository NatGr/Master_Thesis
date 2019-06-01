"""wideresnet model in tf.keras for 32*32 inputs"""
from tensorflow.keras.layers import BatchNormalization, Conv2D, AveragePooling2D, Dense, Activation, Flatten, Dropout, \
    Add
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import VarianceScaling
from .commons import se_block


def build_wrn(inputs, depth, regularizer, num_classes=10, drop_rate=0.0, se_factor=0, channels_dict=None, width=None):
    """builds a wideresnet model given a channels_dict
    if channels_dict is unspecified, width is used instead"""
    assert ((depth - 4) % 6 == 0)  # 4 = the initial conv layer + the 3 conv1*1 when we change the width
    n = int((depth - 4) / 6)

    first_layer_name = "Conv_0"

    if channels_dict is None:
        assert (width is not None)
        channels_dict = {"Conv_0": 16}
        for i in range(1, 4):
            channels_dict[f"Skip_{i}"] = width
            for j in range(n):
                channels_dict[f"Conv_{i}_{j}_1"] = width
                channels_dict[f"Conv_{i}_{j}_2"] = width
            width *= 2

    # 1st conv before any network block
    x = Conv2D(channels_dict[first_layer_name], kernel_size=3, padding="same", use_bias=False,
               name=first_layer_name, kernel_initializer=VarianceScaling(scale=2.0, mode='fan_out'),
               kernel_regularizer=regularizer)(inputs)

    # 1st block
    x = wrn_sub_network(x, 1, n, 1, drop_rate, channels_dict, regularizer=regularizer, se_factor=se_factor)
    # 2nd block
    x = wrn_sub_network(x, 2, n, 2, drop_rate, channels_dict, regularizer=regularizer, se_factor=se_factor)
    # 3rd block
    x = wrn_sub_network(x, 3, n, 2, drop_rate, channels_dict, regularizer=regularizer, se_factor=se_factor)
    # global average pooling and classifier
    x = BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    outputs = Dense(units=num_classes, activation=None,
                    kernel_initializer=VarianceScaling(scale=1/3, mode='fan_in', distribution="uniform"),
                    kernel_regularizer=regularizer, bias_regularizer=regularizer)(x)

    return Model(inputs=inputs, outputs=outputs)


def wrn_sub_network(x, subnet_id, nb_layers, stride, drop_rate, channels_dict, regularizer, se_factor):
    """creates a w.r.n subnetwork with the given parameters, x denotes the input, the output is returned"""
    for i in range(nb_layers):
        conv1_name = f"Conv_{subnet_id}_{i}_1"
        if conv1_name not in channels_dict or channels_dict[conv1_name] == 0:
            if i == 0:  # in the case the first layer was skipped, we have to use the skip 1*1 convolution as
                # input for the second layer
                skip_name = f"Skip_{subnet_id}"
                x = BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(x)
                x = Activation('relu')(x)
                x = Conv2D(channels_dict[skip_name], kernel_size=1, padding="same", use_bias=False, strides=stride,
                           name=skip_name, kernel_initializer=VarianceScaling(scale=2.0, mode='fan_out'),
                           kernel_regularizer=regularizer)(x)

            # otherwise there is nothing to add to the WRN
        else:
            x = wrn_block(x, subnet_id, i, stride if i == 0 else 1, drop_rate, channels_dict, regularizer=regularizer,
                          se_factor=se_factor)

    return x


def wrn_block(x, subnet_id, block_offset, stride, drop_rate, channels_dict, regularizer, se_factor):
    """creates a w.r.n block with the given parameters, the output is returned"""
    conv1_name = f"Conv_{subnet_id}_{block_offset}_1"
    conv2_name = f"Conv_{subnet_id}_{block_offset}_2"

    x_bn_a_1 = BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(x)
    x_bn_a_1 = Activation('relu')(x_bn_a_1)
    out = Conv2D(channels_dict[conv1_name], kernel_size=3, padding="same", use_bias=False, strides=stride,
                 name=conv1_name, kernel_initializer=VarianceScaling(scale=2.0, mode='fan_out'),
                 kernel_regularizer=regularizer)(x_bn_a_1)

    if drop_rate > 0:
        out = Dropout(rate=drop_rate)(out)

    out = BatchNormalization(beta_regularizer=regularizer, gamma_regularizer=regularizer)(out)
    out = Activation('relu')(out)
    out = Conv2D(channels_dict[conv2_name], kernel_size=3, padding="same", use_bias=False, strides=1,
                 name=conv2_name, kernel_initializer=VarianceScaling(scale=2.0, mode='fan_out'),
                 kernel_regularizer=regularizer)(out)

    if block_offset == 0:  # skip_layer
        skip_name = f"Skip_{subnet_id}"
        x = Conv2D(channels_dict[skip_name], kernel_size=1, padding="same", use_bias=False, strides=stride,
                   name=skip_name, kernel_initializer=VarianceScaling(scale=2.0, mode='fan_out'),
                   kernel_regularizer=regularizer)(x_bn_a_1)

    if se_factor != 0 and stride == 1:
        x = se_block(x, channels_dict[conv2_name], se_factor, regularizer)

    return Add()([out, x])
