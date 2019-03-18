"""This script just trains models from scratch, to later be pruned
@Author: Nathan Greffe"""

import argparse
import torch.optim.lr_scheduler as lr_scheduler

from models.wideresnet import *
from functs import *

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--save_file', default='saveto', type=str, help='save file for checkpoints')
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
parser.add_argument('--net', choices=['res'], default='res')

# Net specific
parser.add_argument('--depth', '-d', default=40, type=int, metavar='D', help='depth of wideresnet/densenet')
parser.add_argument('--width', '-w', default=2.0, type=float, metavar='W', help='width of wideresnet')


if not os.path.exists('checkpoints/'):
    os.makedirs('checkpoints/')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if args.net == 'res':
    model = WideResNet(args.depth, args.width, device)
else:
    raise ValueError('pick a valid net')

error_history = []
model.to(device)

# base datasets
full_train_loader, val_loader = get_full_train_val(args.data_loc, args.workers, args.batch_size)


if __name__ == '__main__':

    filename = os.path.join('checkpoints', f'{args.save_file}.t7')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.lr_type == "multistep":
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.epoch_step, gamma=args.lr_decay_ratio)
    elif args.lr_type == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.no_epochs)
    else:
        raise ValueError('pick a valid learning rate type')

    for epoch in range(1, args.no_epochs + 1):
        scheduler.step()

        print(f"Epoch {epoch} -- lr is {optimizer.param_groups[0]['lr']}:")

        train(model, optimizer, full_train_loader, criterion, device)
        # evaluate on validation set
        error_history.append(validate(model, val_loader, criterion, device))

        if epoch == args.no_epochs:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'error_history': error_history,
            }, filename)
