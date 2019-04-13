"""functions used to create smaller models that are to be used in the tables in tensorflow"""
from tensorflow.keras.layers import BatchNormalization, Conv2D, AveragePooling2D, Dense, Activation, Flatten
from tensorflow.keras.models import Model


def make_conv_model(inputs, out_channels, stride):
    """creates a small sequential model composed of a convolution, a batchnorm and a relu activation"""
    outputs = Conv2D(out_channels, kernel_size=3, strides=stride, padding="same", use_bias=False)(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation('relu')(outputs)
    return Model(inputs=inputs, outputs=outputs)


def make_fc_model(inputs, num_classes, width):
    """creates a small sequential model composed of an average pooling and a fully connected layer"""
    outputs = AveragePooling2D(pool_size=width)(inputs)
    outputs = Flatten()(outputs)
    outputs = Dense(units=num_classes)(outputs)
    return Model(inputs=inputs, outputs=outputs)
