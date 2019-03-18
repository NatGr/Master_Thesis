"""file containing useful functions for training and pruning"""

import torch
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as torch_fct


from sklearn.model_selection import train_test_split

import numpy as np
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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

    full_train_set = torchvision.datasets.CIFAR10(root=data_loc, train=True, download=True,
                                                  transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root=data_loc, train=False, download=True, transform=transform_val)

    full_train_loader = torch.utils.data.DataLoader(full_train_set, batch_size=batch_size, shuffle=True,
                                                    num_workers=workers,
                                                    pin_memory=False)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=50, shuffle=False,
                                             num_workers=workers,
                                             pin_memory=False)
    return full_train_loader, val_loader


def get_train_holdout(data_loc, workers, batch_size):
    """get train and holdout set, to perform pruning as in the NetAdapt paper
    :param data_loc: the location of the cifar dataset on disk
    :param workers: the number of workers the data loaders have to use
    :param batch_size: the batch size to use for training
    :return (train_loader, holdout_loader): the train set and holdout set loaders"""
    tmp = torchvision.datasets.CIFAR10(root=data_loc, train=True, download=True, transform=transforms.ToTensor())
    size = tmp.__len__()
    full_train = np.zeros((size, 3, 32, 32))
    full_train_labels = np.zeros(size)

    for i in range(size):
        img_and_label = tmp.__getitem__(i)
        full_train[i, :, :, :] = img_and_label[0].numpy()
        full_train_labels[i] = img_and_label[1]

    x_train, x_holdout, y_train, y_holdout = train_test_split(full_train, full_train_labels, test_size=0.1,
                                                              stratify=full_train_labels)
    train_mean = np.mean(x_train, axis=(0, 2, 3))  # so that we don't use the mean/std of the holdout set
    train_std = np.std(x_train, axis=(0, 2, 3))

    norm_transform = transforms.Normalize(train_mean, train_std)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch_fct.pad(x.unsqueeze(0), [4, 4, 4, 4], mode='reflect').squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm_transform,
    ])

    train_set = torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train), transform=transform_train)
    holdout_set = torch.utils.data.TensorDataset(torch.Tensor(x_holdout), torch.Tensor(y_holdout),
                                                 transform=norm_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=False)
    holdout_loader = torch.utils.data.DataLoader(holdout_set, batch_size=50, shuffle=False,
                                                 num_workers=workers, pin_memory=False)

    return train_loader, holdout_loader


def prune():
    """prunes the model"""


def validate(model, val_loader, criterion, device):
    """ validation of the model
    :param model: the model of the nn
    :param val_loader: the DataLoader of the validation data
    :param criterion: the criterion used to compute the loss
    :param device: the device used to train the network
    :return: the average top1error over the validation
    """
    losses = AverageMeter()
    errors = AverageMeter()

    # switch to evaluate mode
    model.eval()

    begin = time.time()

    for i, (batch_in, batch_target) in enumerate(val_loader):

        batch_in, batch_target = batch_in.to(device), batch_target.to(device)

        # compute output
        output = model(batch_in)

        loss = criterion(output, batch_target)

        # measure accuracy and record loss
        error = 1 - (output == batch_target).sum() / batch_target.size(0)

        losses.update(loss.item(), batch_in.size(0))
        errors.update(error.item(), batch_in.size(0))

    print(f' * fine tune * Elapsed seconds {time.time() - begin :.1f} \t '
          f'Loss {losses.avg :.3f} \t Error {errors.avg :.3f}')

    return errors.avg


def finetune(model, optimizer, criterion, no_steps, dataloader, device):
    """fine tunes the given model
    :param model: the model to fine tune
    :param optimizer: the optimizer to use to optimize the model
    :param criterion: the criterion to compute the loss
    :param no_steps: the number of steps to fine tune
    :param dataloader: the pytoch dataloader to get the training samples from
    :param device: the pytorch device to perform computations on
    """
    model.train()  # switch to train mode
    begin = time.time()
    dataiter = iter(dataloader)
    losses = AverageMeter()
    errors = AverageMeter()

    for i in range(no_steps):

        try:
            batch_in, batch_target = dataiter.next()
        except StopIteration:
            dataiter = iter(dataloader)
            batch_in, batch_target = dataiter.next()

        batch_in, batch_target = batch_in.to(device), batch_target.to(device)

        # compute output
        output = model(batch_in)

        loss = criterion(output, batch_target)

        # measure accuracy and record loss
        error = 1 - (output == batch_target).sum() / batch_target.size(0)

        losses.update(loss.item(), batch_in.size(0))
        errors.update(error.item(), batch_in.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f' * fine tune * Elapsed seconds {time.time() - begin :.1f} \t '
          f'Loss {losses.avg :.3f} \t Error {errors.avg :.3f}')


def train(model, optimizer, train_loader, criterion, device):
    """ training of the model for one epoch
    :param model: the model of the nn
    :param optimizer: the optimizer to use to optimize the model
    :param train_loader: the DataLoader of the validation data
    :param criterion: the criterion used to compute the loss
    :param device: the device used to train the network
    """
    losses = AverageMeter()
    errors = AverageMeter()

    # switch to train mode
    model.train()

    begin = time.time()

    for i, (batch_in, batch_target) in enumerate(train_loader):
        batch_in, batch_target = batch_in.to(device), batch_target.to(device)

        # compute output
        output = model(batch_in)

        loss = criterion(output, batch_target)

        # measure accuracy and record loss
        error = 1 - (output == batch_target).sum() / batch_target.size(0)

        losses.update(loss.item(), batch_in.size(0))
        errors.update(error.item(), batch_in.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f' * fine tune * Elapsed seconds {time.time() - begin :.1f} \t '
          f'Loss {losses.avg :.3f} \t Error {errors.avg :.3f}')
