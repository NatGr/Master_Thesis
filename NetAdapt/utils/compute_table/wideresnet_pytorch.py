"""functions used to create smaller models that are to be used in the tables in pytorch (resnet is the only one for
which there is a pytorch version)"""
import torch
import torch.nn as nn


def make_conv_model(in_channels, out_channels, stride, device):
    """creates a small sequential model composed of a convolution, a batchnorm and a relu activation
    the model is set to eval mode since it is used to measure evaluation time"""
    model = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    model.to(device)
    model.eval()
    return model


def make_fc_model(in_channels, num_classes, width, device):
    """creates a small sequential model composed of an average pooling and a fully connected layer
    the model is set to eval mode since it is used to measure evaluation time"""

    class Flatten(nn.Module):  # not defined in pytorch
        def forward(self, x):
            return x.view(x.size(0), -1)

    model = nn.Sequential(
        nn.AvgPool2d(width),
        Flatten(),
        nn.Linear(in_channels, num_classes)
    )
    model.to(device)
    model.eval()
    return model
