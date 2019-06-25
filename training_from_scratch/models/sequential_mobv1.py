import tensorflow as tf


def build_sequential_mobv1(inputs_shape, blocks_per_subnet, channels_per_subnet, regularizer):
    """builds a mobilenetv1 model using the sequential API"""
    layers = [tf.keras.layers.Conv2D(16, kernel_size=3, activation="relu", padding="same", use_bias=False,
                                     kernel_regularizer=regularizer, input_shape=inputs_shape)]

    for i in range(3):
        for j in range(blocks_per_subnet[i]):
            layers.append(tf.keras.layers.BatchNormalization(beta_regularizer=regularizer,
                                                             gamma_regularizer=regularizer, fused=False))
            layers.append(tf.keras.layers.DepthwiseConv2D(kernel_size=3, activation="relu",
                                                          strides=2 if i != 0 and j == 0 else 1, padding="same",
                                                          use_bias=False, kernel_regularizer=regularizer))
            layers.append(tf.keras.layers.BatchNormalization(beta_regularizer=regularizer,
                                                             gamma_regularizer=regularizer, fused=False))
            layers.append(tf.keras.layers.Conv2D(channels_per_subnet[i], kernel_size=3, activation="relu",
                                                 padding="same", use_bias=False, kernel_regularizer=regularizer))

    layers.append(tf.keras.layers.AveragePooling2D(pool_size=8))
    layers.append(tf.keras.layers.Flatten())
    layers.append(tf.keras.layers.Dense(units=10, activation=None, kernel_regularizer=regularizer,
                                        bias_regularizer=regularizer))

    return tf.keras.Sequential(layers)
