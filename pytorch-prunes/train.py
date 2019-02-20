"""This script just trains models from scratch, to later be pruned"""

import argparse
import json
import os
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from models import *
from funcs import *
import pickle


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--save_file', default='saveto', type=str, help='save file for checkpoints')
parser.add_argument('--base_file', default='bbb', type=str, help='base file for checkpoints')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--data_loc', default='~/Documents/CIFAR-10')

# Learning specific arguments
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-lr', '--learning_rate', default=.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='multistep', type=str, help='learning rate strategy (default: cosine)',
                    choices=['cosine', 'multistep'])
parser.add_argument('-epochs', '--no_epochs', default=200, type=int, metavar='epochs', help='no. epochs')
parser.add_argument('--epoch_step', default='[60,120,160]', type=str, help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay factor')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float, metavar='W', help='weight decay')
parser.add_argument('--eval', '-e', action='store_true', help='resume from checkpoint')
parser.add_argument('--prune_train_prune', action='store_true', help='to use when performing alternations of training '
                                                                     'from scratch and pruning')
parser.add_argument('--mask', '-m', type=int, help='mask mode', default=0)
parser.add_argument('--deploy', '-de', action='store_true', help='deploy a model whose structure was previously pruned')
parser.add_argument('--params_left', '-pl', default=0, type=int, help='prune til...')
parser.add_argument('--net', choices=['res', 'dense', 'notsodense1'], default='res')
parser.add_argument('--save_every', default=50, type=int, help='save model every X EPOCHS')
parser.add_argument('--list_channels', default=None, help='pickle file containing the number of channels to use for '
                                                          'every layer in the network as a python list')
parser.add_argument('--channels_factor', default=1., type=float, help='float by which to multiply the number of '
                                                                      'channels')

# Net specific
parser.add_argument('--depth', '-d', default=40, type=int, metavar='D', help='depth of wideresnet/densenet')
parser.add_argument('--width', '-w', default=2.0, type=float, metavar='W', help='width of wideresnet')
parser.add_argument('--growth', default=12, type=int, help='growth rate of densenet')
parser.add_argument('--transition_rate', default=0.5, type=float, help='transition rate of densenet')
parser.add_argument('--fast_train', '-ft', action='store_true', help='trains the denseNet faster at the cost of '
                                                                    'more memory')
parser.add_argument('--notsodense_k', default=3, type=int, help='the number of previous layers whose output will '
                                                                'be used as input for the next layer in a block')

# Uniform bottlenecks
parser.add_argument('--bottle', action='store_true', help='Linearly scale bottlenecks')
parser.add_argument('--bottle_mult', default=0.5, type=float, help='bottleneck multiplier')


if not os.path.exists('checkpoints/'):
    os.makedirs('checkpoints/')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# loads ans scale the number of bottleneck channels per layer
list_channels = None
if args.list_channels is not None:
    with open(os.path.join('nbr_channels', f"{args.list_channels}.pickle"), 'rb') as file:
        list_channels = pickle.load(file)[0]
        list_channels = [i*args.channels_factor for i in list_channels]


if args.net == 'res':
    if not args.bottle:
        model = WideResNet(args.depth, args.width, mask=args.mask)
    else:
        model = WideResNetBottle(args.depth, args.width, mid_channels=args.bottle_mult if list_channels is None else
        list_channels)
elif args.net == 'dense':
    if not args.bottle:
        model = DenseNet(args.growth, args.depth, args.transition_rate, 10, True, mask=args.mask,
                         efficient=not args.fast_train)
    else:
        model = DenseNet(args.growth, args.depth, args.transition_rate, 10, True, mid_channels=args.bottle_mult if
        list_channels is None else list_channels, efficient=not args.fast_train)
elif args.net == 'notsodense1':
    if not args.bottle:
        model = NotSoDenseNet1(args.growth, args.depth, args.transition_rate, 10, True, args.notsodense_k,
                              mask=args.mask, efficient=not args.fast_train)
    else:
        model = NotSoDenseNet1(args.growth, args.depth, args.transition_rate, 10, True, args.notsodense_k,
                              mid_channels=args.bottle_mult if list_channels is None else list_channels,
                              efficient=not args.fast_train)
else:
    raise ValueError('pick a valid net')

pruner = Pruner([])
error_history = []
param_history = []
prune_history = []
pruning_epoch = 0

if args.deploy or args.prune_train_prune:
    # Feed example to activate masks
    model(torch.rand(1, 3, 32, 32))
    state = torch.load('checkpoints/%s.t7' % args.base_file)

    if 'prune_history' in state:  # we are using a model that has been pruned
        pruner = Pruner([])
        pruner._get_masks(model)
        for ii in state['prune_history']:
            pruner.fixed_prune(model, ii, verbose=False)
        pruning_epoch = state["epoch"]
        error_history = state['error_history']
        prune_history = state['prune_history']
        param_history = state['param_history']

if not args.prune_train_prune:  # in the prune_train_prune settings, we do not want to compress the model because it
    # would render prune_history invalid since it would remove some channels from the network and thus change the
    # channels offsets
    pruner.compress(model)

if (args.deploy or args.prune_train_prune) and args.eval:
    model.load_state_dict(state['state_dict'])

get_inf_params(model)
time.sleep(1)
model.to(device)

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])  # CIFAR-10 train set mean and std for
# each channel

print('==> Preparing data..')
num_classes = 10

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                      [4, 4, 4, 4], mode='reflect').squeeze()),
    transforms.ToPILImage(),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    normalize,

])

trainset = torchvision.datasets.CIFAR10(root=args.data_loc,
                                        train=True, download=True, transform=transform_train)
valset = torchvision.datasets.CIFAR10(root=args.data_loc,
                                      train=False, download=True, transform=transform_val)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.workers,
                                          pin_memory=False)
valloader = torch.utils.data.DataLoader(valset, batch_size=50, shuffle=False,
                                        num_workers=args.workers,
                                        pin_memory=False)

epoch_step = json.loads(args.epoch_step)


def train():
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    begin = time.time()

    for i, (batch_in, batch_target) in enumerate(trainloader):
        batch_in, batch_target = batch_in.to(device), batch_target.to(device)

        # compute output
        output = model(batch_in)

        loss = criterion(output, batch_target)

        # measure accuracy and record loss
        err1 = get_error(output.detach(), batch_target)[0]

        losses.update(loss.item(), batch_in.size(0))
        top1.update(err1.item(), batch_in.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i != 0 and i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} (avg: {loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} (avg: {top1.avg:.3f})'.format(
                    epoch, i, len(trainloader), loss=losses, top1=top1))

    print(' * train * Elapsed seconds {elapsed:.1f} \t Loss {loss.avg:.3f} \t Error@1 {top1.avg:.3f}'
          .format(top1=top1, loss=losses, elapsed=time.time() - begin))


if __name__ == '__main__':

    filename = 'checkpoints/%s.t7' % args.save_file
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.lr_type == "multistep":
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=epoch_step, gamma=args.lr_decay_ratio)
    elif args.lr_type == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.no_epochs)
    else:
        raise ValueError('pick a valid learning rate type')

    if not args.eval:

        for epoch in range(args.no_epochs):
            scheduler.step()

            print('Epoch %d:' % epoch)
            print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])
            # train for one epoch
            train()
            # evaluate on validation set
            validation_error = validate(model, valloader, criterion, device, args.print_freq)

            if not args.prune_train_prune:
                error_history.append(validation_error)
                if epoch != 0 and ((epoch % args.save_every == 0) or (epoch + 1 == args.no_epochs)):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'error_history': error_history,
                    }, filename=filename)

            else:  # retrains a previously pruned model, in this case, the only element we will add to error_history is
                # our error at convergence
                if epoch + 1 == args.no_epochs:
                    error_history.append(validation_error)
                    param_history.append(param_history[-1])
                if epoch != 0 and ((epoch % args.save_every == 0) or (epoch + 1 == args.no_epochs)):
                    save_checkpoint({
                        'epoch': pruning_epoch,
                        'state_dict': model.state_dict(),
                        'error_history': error_history,
                        'param_history': param_history,
                        'prune_history': prune_history,
                    }, filename=filename)

    else:
        if not args.deploy:
            model.load_state_dict(torch.load(filename)['state_dict'])
        epoch = 0
        validate(model, valloader, criterion, device, args.print_freq)
