"""wideresnet model in tf.keras"""
from tensorflow.keras.layers import BatchNormalization, Conv2D, AveragePooling2D, Dense, Activation, Flatten
from tensorflow.keras.models import Model


def build_wrn(inputs, depth, channels_dict, num_classes=10, drop_rate=0.0):
	"""builds a wideresnet model given a channels_dict"""

	assert ((depth - 4) % 6 == 0)  # 4 = the initial conv layer + the 3 conv1*1 when we change the width
    n = (depth - 4) / 6

    first_layer_name = "Conv_0"

    n_channels = [channels_dict[first_layer_name], channels_dict["Skip_1"],
                  channels_dict["Skip_2"], channels_dict["Skip_3"]]

    # 1st conv before any network block
    x = Conv2D(n_channels[0], kernel_size=3, padding='SAME', use_bias=False)(inputs)

    # 1st block
    self.block1 = WideResNetSubNetwork(1, n, n_channels[0], n_channels[1], 1, drop_rate, channels_dict)
    # 2nd block
    self.block2 = WideResNetSubNetwork(2, n, n_channels[1], n_channels[2], 2, drop_rate, channels_dict)
    # 3rd block
    self.block3 = WideResNetSubNetwork(3, n, n_channels[2], n_channels[3], 2, drop_rate, channels_dict)
    # global average pooling and classifier
    self.bn1 = nn.BatchNorm2d(n_channels[3])
    self.relu = nn.ReLU(inplace=True)
    self.avg_pool = nn.AvgPool2d(8)
    self.fc = nn.Linear(n_channels[3], num_classes)
    self.nChannels = n_channels[3]

    return Model(inputs=inputs, outputs=output)