"""Pruning script
@Author: Nathan Greffe"""

import argparse

from models.wideresnet import *
from functs import *

parser = argparse.ArgumentParser(description='Pruning')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--save_file', default='wrn16_2_p', type=str, help='save file for checkpoints')
parser.add_argument('--data_loc', default='~/Documents/CIFAR-10', type=str, help='where is the dataset')

# Learning specific arguments
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-lr', '--learning_rate', default=8e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-epochs', '--no_epochs', default=1300, type=int, metavar='epochs', help='no. epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float, metavar='W', help='weight decay')
parser.add_argument('--short_term_fine_tune', default=100, type=int, help='prune every X steps')
parser.add_argument('--long_term_fine_tune', default=100, type=int, help='long term fine tune on the whose dataset')
parser.add_argument('--base_model', default='base_model', type=str, help='basemodel (in folder checkpoints)')
parser.add_argument('--net', default='res', type=str, help='network architecture')
parser.add_argument('--width', default=2.0, type=float, metavar='D')
parser.add_argument('--depth', default=40, type=int, metavar='W')


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if args.net == 'res':
    model = WideResNet(args.depth, args.width, device)
else:
    raise ValueError('pick a valid net')

error_history = []
prune_history = []
model.load_state_dict(torch.load(os.path.join('checkpoints', f'{args.base_model}.t7'),
                                 map_location='cpu')['state_dict'], strict=True)
model.to(device)

print('==> Preparing data..')

# base datasets
full_train_loader, val_loader = get_full_train_val(args.data_loc, args.workers, args.batch_size)

# dataset split into train/holdout set
train_loader, holdout_loader = get_train_holdout(args.data_loc, args.workers, args.batch_size)


if __name__ == '__main__':

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.no_epochs+1):

        print(f"Epoch {epoch} -- lr is {optimizer.param_groups[0]['lr']}:")

        # Prune
        prune_history.append(prune())

        # fine tune for one epoch
        finetune(model, optimizer, criterion, args.short_term_fine_tune, train_loader, device)

        # evaluate on validation set
        error_history.append(validate(model, val_loader, criterion, device))

    # long term fine tune
    finetune(model, optimizer, criterion, args.long_term_fine_tune, full_train_loader, device)
    error_history.append(validate(model, val_loader, criterion, device))

    # Save
    filename = os.path.join('checkpoints', f'{args.save_file}.t7')
    torch.save({
        'epoch': args.no_epochs,
        'state_dict': model.state_dict(),
        'error_history': error_history,
        'prune_history': prune_history,
    }, filename)



