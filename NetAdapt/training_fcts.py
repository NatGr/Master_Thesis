"""file containing useful functions for training and pruning
@Author: Nathan Greffe"""

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


def validate(model, val_loader, criterion, device, val_set_name="val", memory_leak=False):
    """ validation of the model
    :param model: the model of the nn
    :param val_loader: the DataLoader of the validation data
    :param criterion: the criterion used to compute the loss
    :param device: the device used to train the network
    :param val_set_name: name to display on the print function to indicate which dataset we use for validation
    :param memory_leak: in pytorch 1.0.1.post2, I get a memory leak here unless I call loss.backward() at the end, I
    have no idea why and this isn't the only weird memory problem I got
    :return: the average top1error over the validation
    """
    losses = AverageMeter()
    errors = AverageMeter()

    # switch to evaluate mode
    model.eval()

    begin = time.time()

    for batch_in, batch_target in val_loader:

        batch_in, batch_target = batch_in.to(device), batch_target.to(device)

        # compute output
        output = model(batch_in)

        loss = criterion(output, batch_target)

        # measure accuracy and record loss
        error = get_error(output, batch_target)

        losses.update(loss.item(), batch_in.size(0))
        errors.update(error.item(), batch_in.size(0))

    if memory_leak:  # see function docstring
        loss.backward()

    print(f' * {val_set_name}\t* Elapsed seconds {time.time() - begin :.1f} \t '
          f'Loss {losses.avg :.3f} \t Error {errors.avg :.3f}')

    return errors.avg


def finetune(model, optimizer, criterion, no_steps, dataloader, layer, device):
    """fine tunes the given model
    :param model: the model to fine tune
    :param optimizer: the optimizer to use to optimize the model
    :param criterion: the criterion to compute the loss
    :param no_steps: the number of steps to fine tune
    :param dataloader: the pytoch dataloader to get the training samples from
    :param device: the pytorch device to perform computations on
    :param layer: the name of the layer that was just pruned
    """
    model.reset_fisher()  # resets the accumulated fisher values to 0 since we fine_tune between pruning ops
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
        error = get_error(output, batch_target)

        losses.update(loss.item(), batch_in.size(0))
        errors.update(error.item(), batch_in.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f' * fine tune after pruning {layer} * Elapsed seconds {time.time() - begin :.1f} \t '
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
        error = get_error(output, batch_target)

        losses.update(loss.item(), batch_in.size(0))
        errors.update(error.item(), batch_in.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f' * train\t* Elapsed seconds {time.time() - begin :.1f} \t '
          f'Loss {losses.avg :.3f} \t Error {errors.avg :.3f}')


def get_error(output, target):
    """given the output of the NN and the target labels, return the error"""
    _, output = output.max(1)  # output is one-hot encoded
    output = output.view(-1)
    return 100 - output.eq(target).float().sum().mul(100 / target.size(0))
