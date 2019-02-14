"""Pruning script"""

import argparse
import os

import torch.utils.model_zoo as model_zoo

from funcs import *
from models import *


parser = argparse.ArgumentParser(description='Pruning')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--GPU', default='0', type=str, help='GPU to use')
parser.add_argument('--save_file', default='wrn16_2_p', type=str, help='save file for checkpoints')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint or pruned model retrained '
                                                                'from scratch')
parser.add_argument('--resume_ckpt', default='checkpoint', type=str,
                    help='save file for resumed checkpoint')
parser.add_argument('--data_loc', default='~/Documents/CIFAR-10', type=str, help='where is the dataset')

# Learning specific arguments
parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd', type=str, help='optimizer')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-lr', '--learning_rate', default=8e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-epochs', '--no_epochs', default=1300, type=int, metavar='epochs', help='no. epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float, metavar='W', help='weight decay')
parser.add_argument('--prune_every', default=100, type=int, help='prune every X steps')
parser.add_argument('--save_every', default=100, type=int, help='save model every X EPOCHS')
parser.add_argument('--random', default=False, type=bool, help='Prune at random')
parser.add_argument('--base_model', default='base_model', type=str, help='basemodel')
parser.add_argument('--val_every', default=1, type=int, help='val model every X EPOCHS')
parser.add_argument('--mask', default=1, type=int, help='Mask type')
parser.add_argument('--l1_prune', default=False, type=bool, help='Prune via l1 norm')
parser.add_argument('--net', default='dense', type=str, help='dense, res')
parser.add_argument('--width', default=2.0, type=float, metavar='D')
parser.add_argument('--depth', default=40, type=int, metavar='W')
parser.add_argument('--growth', default=12, type=int, help='growth rate of densenet')
parser.add_argument('--transition_rate', default=0.5, type=float, help='transition rate of densenet')

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

device = torch.device("cuda:%s" % '0' if torch.cuda.is_available() else "cpu")


if args.net == 'res':
    model = WideResNet(args.depth, args.width, mask=args.mask)
elif args.net == 'dense':
    model = DenseNet(args.growth, args.depth, args.transition_rate, 10, True, mask=args.mask)
else:
    raise ValueError('pick a valid net')

if args.resume:
    # Feed example to activate masks
    model(torch.rand(1, 3, 32, 32))
    state = torch.load('checkpoints/%s.t7' % args.resume_ckpt, map_location='cpu')

    error_history = state['error_history']
    prune_history = state['prune_history']
    param_history = state['param_history']
    start_epoch = state['epoch']

    pruner = Pruner([])
    pruner._get_masks(model)
    for ii in state['prune_history']:
        pruner.fixed_prune(model, ii)
    model.load_state_dict(state['state_dict'], strict=True)

else:
    error_history = []
    prune_history = []
    param_history = []
    start_epoch = 0
    model.load_state_dict(torch.load('checkpoints/%s.t7' % args.base_model, map_location='cpu')['state_dict'],
                          strict=True)

model.to(device)

normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]
normTransform = transforms.Normalize(normMean, normStd)

print('==> Preparing data..')
num_classes = 10

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normTransform
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    normTransform
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

pruner = Pruner(prune_history)

NO_STEPS = args.prune_every


def finetune():
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    begin = time.time()

    dataiter = iter(trainloader)

    for i in range(0, NO_STEPS):

        try:
            batch_in, batch_target = dataiter.next()
        except StopIteration:
            dataiter = iter(trainloader)
            batch_in, batch_target = dataiter.next()

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
            print('Prunepoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} (avg: {loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} (avg: {top1.avg:.3f})'.format(
                    epoch, i, NO_STEPS, loss=losses, top1=top1))

    print(' * fine tune * Elapsed seconds {elapsed:.1f} \t Loss {loss.avg:.3f} \t Error@1 {top1.avg:.3f}'
          .format(top1=top1, loss=losses, elapsed=time.time() - begin))


def prune():
    if args.random is False:
        if args.l1_prune is False:
            print('fisher pruning')
            pruner.fisher_prune(model, prune_every=args.prune_every)
        else:
            print('l1 pruning')
            pruner.l1_prune(model, prune_every=args.prune_every)
    else:
        print('random pruning')
        pruner.random_prune(model, )


if __name__ == '__main__':

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.no_epochs):

        print('Epoch %d:' % epoch)
        print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])

        # fine tune for one epoch
        finetune()

        # evaluate on validation set
        if epoch != 0 and ((epoch % args.val_every == 0) or (epoch + 1 == args.no_epochs)):  # Save at last epoch!
            error_history.append(validate(model, valloader, criterion, device, args.print_freq))
            no_params = pruner.get_cost(model) + model.fixed_params
            print(f"{no_params / 1e6 :.2f} M parameters remaining")
            param_history.append(no_params)

        # Prune
        prune()

        # Save before pruning
        if epoch != 0 and ((epoch % args.save_every == 0) or (epoch + 1 == args.no_epochs)):
            filename = 'checkpoints/%s_%d_prunes.t7' % (args.save_file, epoch)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'error_history': error_history,
                'param_history': param_history,
                'prune_history': pruner.prune_history,
            }, filename=filename)
