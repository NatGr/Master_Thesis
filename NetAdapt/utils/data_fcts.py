"""file containing useful functions and classes to handle data loading
@Author: Nathan Greffe"""

import torch
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as torch_fct
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import numpy as np


class TensorDataSetWithAugmentation(Dataset):
    """subclass of Dataset than handles data augmentation for tensors"""
    def __init__(self, x_tensor, y_tensor, transform=None):
        super(TensorDataSetWithAugmentation, self).__init__()
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.transform = transform

    def __getitem__(self, item):
        x = self.x_tensor[item, :, :, :]
        y = self.y_tensor[item]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return self.x_tensor.size(0)


def get_full_train_val(data_loc, workers, batch_size):
    """get full train and validation set
    :param data_loc: the location of the cifar dataset on disk
    :param workers: the number of workers the data loaders have to use
    :param batch_size: the batch size to use for training
    :return (full_train_loader, val_loader): the train set and holdout set loaders"""
    norm_mean = [0.49139968, 0.48215827, 0.44653124]
    norm_std = [0.24703233, 0.24348505, 0.26158768]
    norm_transform = transforms.Normalize(norm_mean, norm_std)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch_fct.pad(x.unsqueeze(0), [4, 4, 4, 4], mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm_transform,
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        norm_transform
    ])

    full_train_set = torchvision.datasets.CIFAR10(root=data_loc, train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root=data_loc, train=False, download=True, transform=transform_val)

    full_train_loader = DataLoader(full_train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                                   pin_memory=False)
    val_loader = DataLoader(valset, batch_size=50, shuffle=False, num_workers=workers, pin_memory=False)

    return full_train_loader, val_loader


def get_train_holdout(data_loc, workers, batch_size, holdout_prop):
    """get train and holdout set, to perform pruning as in the NetAdapt paper
    :param data_loc: the location of the cifar dataset on disk
    :param workers: the number of workers the data loaders have to use
    :param batch_size: the batch size to use for training
    :param holdout_prop: fraction of the total dataset that will end up in holdout set
    :return (train_loader, holdout_loader): the train set and holdout set loaders"""
    tmp = torchvision.datasets.CIFAR10(root=data_loc, train=True, download=True, transform=transforms.ToTensor())
    size = tmp.__len__()
    full_train = np.zeros((size, 3, 32, 32), dtype=np.float32)
    full_train_labels = np.zeros(size, dtype=np.long)

    for i in range(size):
        img_and_label = tmp.__getitem__(i)
        full_train[i, :, :, :] = img_and_label[0].numpy()
        full_train_labels[i] = img_and_label[1]

    x_train, x_holdout, y_train, y_holdout = train_test_split(full_train, full_train_labels, test_size=holdout_prop,
                                                              stratify=full_train_labels, random_state=17)  # so as to
    # always have the same separation since it's used in several files

    train_mean = np.mean(x_train, axis=(0, 2, 3))  # so that we don't use the mean/std of the holdout set
    train_std = np.std(x_train, axis=(0, 2, 3))

    norm_transform = transforms.Normalize(train_mean, train_std)
    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: torch_fct.pad(x.unsqueeze(0), [4, 4, 4, 4], mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm_transform,
    ])

    x_train, x_holdout = torch.from_numpy(x_train), torch.from_numpy(x_holdout)
    y_train, y_holdout = torch.from_numpy(y_train), torch.from_numpy(y_holdout)

    train_set = TensorDataSetWithAugmentation(x_train, y_train, transform=transform_train)
    holdout_set = TensorDataSetWithAugmentation(x_holdout, y_holdout, transform=norm_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False)
    holdout_loader = DataLoader(holdout_set, batch_size=50, shuffle=False, num_workers=workers, pin_memory=False)

    return train_loader, holdout_loader
