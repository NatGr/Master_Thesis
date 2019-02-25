import random
import numpy as np
import torchvision.transforms as transforms
import torchvision
import time
from functools import reduce
from models import *
import random
import time
import operator
import torchvision
import torchvision.transforms as transforms

from models import *


class Pruner:
    def __init__(self, prune_history, module_name='MaskBlock'):
        # First get vector of masks
        self.module_name = module_name
        self.masks = None
        self.prune_history = prune_history

    def fisher_prune(self, model, prune_every):
        """prunes out the channel with the smallest fischer quantity"""
        self._get_fisher(model)
        tot_loss = self.fisher.div(prune_every) + 1e6 * (1 - self.masks)  # dummy value for already removed channels
        _, argmin = torch.min(tot_loss, 0)
        self.prune(model, argmin.item())
        self.prune_history.append(argmin.item())

    def fixed_prune(self, model, id_channel, verbose=True):
        """prunes out the channel with id id_channel"""
        self.prune(model, id_channel, verbose)
        self.prune_history.append(id_channel)

    def random_prune(self, model):
        """prunes out a channel at random"""
        masks = []
        for m in model.modules():
            if m._get_name() == self.module_name:
                masks.append(m.mask.detach())

        self.masks = self.concat(masks)
        masks_on = [i for i, v in enumerate(self.masks) if v == 1]
        random_pick = random.choice(masks_on)
        self.prune(model, random_pick)
        self.prune_history.append(random_pick)

    def l1_prune(self, model):
        """prunes out the channel with smallest l1 norm"""
        masks = []
        l1_norms = []

        for m in model.modules():
            if m._get_name() == 'MaskBlock':
                l1_norm = torch.sum(m.conv1.weight, (1, 2, 3)).detach().cpu().numpy()
                masks.append(m.mask.detach())
                l1_norms.append(l1_norm)

        masks = self.concat(masks)
        self.masks = masks
        l1_norms = np.concatenate(l1_norms)

        l1_norms_on = []
        for m, l in zip(masks, l1_norms):
            if m == 1:
                l1_norms_on.append(l)
            else:
                l1_norms_on.append(9999.)  # dummy value

        smallest_norm = min(l1_norms_on)
        pick = np.where(l1_norms == smallest_norm)[0][0]

        self.prune(model, pick)
        self.prune_history.append(pick)

    def prune(self, model, feat_index, verbose=True):
        """
        feat_index refers to the index of a feature map. This function modifies the mask to turn it off.
        set verbose to False in order to remove the number of channels pruned so far and which channel was pruned
        """

        if verbose:
            print(f'Pruned {len(self.prune_history)} out of {len(self.masks)} channels so far')
        if len(self.prune_history) > len(self.masks):
            raise Exception('Time to stop')

        safe = 0
        running_index = 0
        for m in model.modules():
            if m._get_name() == self.module_name:
                mask_indices = range(running_index, running_index + len(m.mask))

                if feat_index in mask_indices:
                    if verbose:
                        print('Pruning channel %d' % feat_index)
                    local_index = mask_indices.index(feat_index)
                    m.mask[local_index] = 0
                    safe = 1
                    break
                else:
                    running_index += len(m.mask)
                    # print(running_index)
        if safe == 0:
            raise Exception('The provided index doesn''t correspond to any feature maps. This is bad.')

    @staticmethod
    def compress(model, verbose=True):
        """
        removes the pruned channels from the MaskBlocks
        returns, for each layer, a pair containing the number of channels left and the initial number of channels
        """
        channels_left = []
        for m in model.modules():
            if m._get_name() == 'MaskBlock':
                if verbose:
                    print(f"block {len(channels_left)+1:>3}", end=": ")  # starting at 1 to follow the convention of the
                    # paper
                channels_left.append(m.compress_weights(verbose))
        return channels_left

    def _get_fisher(self, model):
        """computes the fisher score of the different layers"""
        masks = []
        fisher = []

        self._update_cost(model)

        for m in model.modules():
            if m._get_name() == self.module_name:
                masks.append(m.mask.detach())
                fisher.append(m.running_fisher.detach())

                # Now clear the fisher cache
                m.reset_fisher()

        self.masks = self.concat(masks)
        self.fisher = self.concat(fisher)

    def _get_masks(self, model):
        """get all the masks (the modules whose class name is self.module_name) of the model"""
        masks = []

        for m in model.modules():
            if m._get_name() == self.module_name:
                masks.append(m.mask.detach())  # detach creates a copy that shares memory with the original

        self.masks = self.concat(masks)

    def _update_cost(self, model):
        """update the number of parameters of the MaskBlocks (changed by the last pruning)"""
        for m in model.modules():
            if m._get_name() == self.module_name:
                m.cost()

    def get_cost(self, model):
        """update the number of parameters of the MaskBlocks and return the sum of these costs over the whole model"""
        params = 0
        for m in model.modules():
            if m._get_name() == self.module_name:
                m.cost()
                params += m.params
        return params

    @staticmethod
    def concat(input):
        """concatenates among first dimension, i.e. concat([zeros(2*2), zeros(2*2)]) -> zeros(4*2)"""
        return torch.cat([item for item in input])


def find(input):
    # Find as in MATLAB to find indices in a binary vector
    return [i for i, j in enumerate(input) if j]


def concat(input):
    return torch.cat([item for item in input])


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def get_error(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res


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


def get_inf_params(net, verbose=True, sd=False):
    if sd:
        params = net
    else:
        params = net.state_dict()
    tot = 0
    conv_tot = 0
    for p in params:
        no = params[p].view(-1).__len__()

        if ('num_batches_tracked' not in p) and ('running' not in p) and ('mask' not in p):
            tot += no

            if verbose:
                print('%s has %d params' % (p, no))
        if 'conv' in p:
            conv_tot += no

    if verbose:
        print('Net has %d conv params' % conv_tot)
        print('Net has %d params in total' % tot)

    return tot


def len_gen(gen):
    return sum(1 for _ in gen)


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return len_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


count_ops = 0
count_params = 0


def measure_model(model, height, width):
    """
    This function should be called on a compressed model!
    this function (and the other ones related) were taken from
    https://github.com/ShichenLiu/CondenseNet/blob/master/utils.py by the authors of
    "Pruning neural networks: is it time to nip it in the bud?", I tried to add comments here and I adapted it so that
    it works but I did not change the structure (altough it's not that clean).

    :param model: the model we want to measure the number of operations and parameters from (assumed to take RGB
    images as input)
    :param height: the height of the images the model would take as input
    :param width: the width of the images the model would take as input
    :return (count_ops, count_params): the number of operations and parameters of the model
    """
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = Variable(torch.zeros(1, 3, height, width))

    def should_measure(x):
        return is_leaf(x)

    def modify_forward(model):
        """
        modifies the forward pass of the model so as to make it compute the number of operations and parameters in
        each layer, what the forward attributes previously contained is stored into old_forward attributes
        """
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)

                    return lambda_forward

                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        # replace the forward attribute by the old_forward attribute for every leaf in model
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)  # performs the modified forward pass that will count the number of parameters and operations
    restore_forward(model)

    return count_ops, count_params


def measure_layer(layer, x):
    """
    measures the number of operations and parameters of layer given input x and add it to the global variables count_ops
    and count_params.
    The input batch size should be 1 to call this function.
    :param layer: the layer whose parameters we are counting (given input x)
    :param x: the input given to the layer
    """
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    # ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)

    # ops_learned_conv
    elif type_name in ['LearnedGroupConv']:
        measure_layer(layer.relu, x)
        measure_layer(layer.norm, x)
        conv = layer.conv
        out_h = int((x.size()[2] + 2 * conv.padding[0] - conv.kernel_size[0]) /
                    conv.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * conv.padding[1] - conv.kernel_size[1]) /
                    conv.stride[1] + 1)
        delta_ops = conv.in_channels * conv.out_channels * conv.kernel_size[0] * \
                    conv.kernel_size[1] * out_h * out_w / layer.condense_factor * multi_add
        delta_params = get_layer_param(conv) / layer.condense_factor

    # ops_nonlinearity
    elif type_name in ['ReLU']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)

    # ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    # ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)

    # ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout']:
        delta_params = get_layer_param(layer)

    # Identity used to compute fischer information, Zero(Make) used when layer gets all his channels pruned;
    # they are to be ignored here
    elif type_name in ['Identity', 'Zero', 'ZeroMake']:
        pass

    # unknown layer type
    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return


def validate(model, valloader, criterion, device, print_freq):
    """
    :param model: the model of the nn
    :param valloader: the DataLoader of the validation data
    :param criterion: the criterion used to compute the loss
    :param device: the device used to train the network
    :param print_freq: the number of batches after which we print infos about the validation
    :return: the average top1error over the validation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    begin = time.time()

    for i, (batch_in, batch_target) in enumerate(valloader):

        batch_in, batch_target = batch_in.to(device), batch_target.to(device)

        # compute output
        output = model(batch_in)

        loss = criterion(output, batch_target)

        # measure accuracy and record loss
        err1 = get_error(output.detach(), batch_target)[0]

        losses.update(loss.item(), batch_in.size(0))
        top1.update(err1.item(), batch_in.size(0))

        if i != 0 and i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} (avg: {loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} (avg: {top1.avg:.3f})'.format(
                    i, len(valloader), loss=losses,
                    top1=top1))

    print(' * val * Elapsed seconds {elapsed:.1f} \t Loss {loss.avg:.3f} \t Error@1 {top1.avg:.3f}'
          .format(top1=top1, loss=losses, elapsed=time.time() - begin))

    # Record Top 1 for CIFAR
    return top1.avg
