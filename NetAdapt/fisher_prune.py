"""Fisher Pruning script (as in pytorch-prunes) that uses the perf-table cost and the network class of NetAdapt
@Author: Nathan Greffe
"""

import argparse

from models.wideresnet import *
from utils.training_fcts import *
from utils.data_fcts import *
import json

parser = argparse.ArgumentParser(description='Pruning')
parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--save_file', default='wrn16_2_p', type=str, help='save file for checkpoints')
parser.add_argument('--data_loc', default='~/Documents/CIFAR-10', type=str, help='where is the dataset')
parser.add_argument('--pruning_fact', type=float, help="proportion of inference time that will approx be pruned away")
parser.add_argument('--base_model', default='base_model', type=str, help='basemodel (in folder checkpoints)')
parser.add_argument('--perf_table', default='res-40-2', type=str, help='the perf_table (assumed to be in perf_tables '
                                                                       'folders and without the extension)')

# Learning specific arguments
parser.add_argument('--net', default='res', type=str, help='network architecture')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--learning_rate', default=8e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight decay')
parser.add_argument('--short_term_fine_tune', default=100, type=int, help='number of batches')
parser.add_argument('--save_several_channels', default='', type=str,
                    help='json list with number of channels prunings after which we save the number of '
                         'channels independently of the reduction objective')

parser.add_argument('--use_partial_train_set', action='store_true',
                    help='performs the pruning on 90% of the training set (to use the same base networks as NetAdapt')
parser.add_argument('--holdout_prop', default=0.1, type=float,
                    help='1 - fraction of training set used for training when use_partial_train_set is set to true')

parser.add_argument('--width', default=2.0, type=float)
parser.add_argument('--depth', default=40, type=int)


args = parser.parse_args()
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


if args.net == 'res':
    model = WideResNet(args.depth, args.width, device)
else:
    raise ValueError('pick a valid net')

error_history = []
prune_history = []
table_costs_history = []
model.load_state_dict(torch.load(os.path.join('checkpoints', f'{args.base_model}.t7'),
                                 map_location='cpu')['state_dict'], strict=True)
model.to(device)
model.load_table(os.path.join("perf_tables", f"{args.perf_table}.pickle"))

print('==> Preparing data..')
if args.use_partial_train_set:
    train_loader, _ = get_train_holdout(args.data_loc, args.workers, args.batch_size, args.holdout_prop)
    _, val_loader = get_full_train_val(args.data_loc, args.workers, args.batch_size)
else:
    train_loader, val_loader = get_full_train_val(args.data_loc, args.workers, args.batch_size)

if args.save_several_channels != '':
    channels_to_save = json.loads(args.save_several_channels)
else:
    channels_to_save = []


if __name__ == '__main__':

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    prune_history.append(None)
    table_costs_history.append(model.total_cost)
    error_history.append(validate(model, val_loader, criterion, device, val_set_name="val"))

    # first fine-tune to have gradients so that we can fisher-prune some layers
    finetune(model, optimizer, criterion, args.short_term_fine_tune, train_loader, "", device)

    red_objective = (1 - args.pruning_fact) * model.total_cost
    step_number = 1

    while model.total_cost > red_objective:

        print(f"Pruning step number {step_number} -- model cost is {model.total_cost :.2f}s")

        # Prune
        pruned_layer = model.prune_one_channel(args.short_term_fine_tune)

        finetune(model, optimizer, criterion, args.short_term_fine_tune, train_loader, "", device)

        error_history.append(validate(model, val_loader, criterion, device, val_set_name="val"))

        prune_history.append((pruned_layer, 1))
        table_costs_history.append(model.total_cost)

        if step_number in channels_to_save:
            print(f"Saving after having pruned {step_number} channels, current error == {error_history[-1]}, "
                  f"current cost == {model.total_cost:.2f}")
            filename2 = os.path.join('nbr_channels', f'{args.save_file}-{step_number}ch.pickle')
            with open(filename2, 'wb') as file:
                pickle.dump(model.num_channels_dict, file)

        step_number += 1

    print(f"pruned network inference time according to perf_table: {model.total_cost :.2f}")

    for layer_name in model.to_prune:
        layer = getattr(self, layer_name, None)
        if layer is None:
            print(f"{layer_name} has been pruned away")
        else:
            print(f"{layer_name} has {layer.out_channels} output channels")

    # Save
    filename = os.path.join('checkpoints', f'{args.save_file}.t7')
    torch.save({
        'epoch': step_number,
        'state_dict': model.state_dict(),
        'error_history': error_history,
        'prune_history': prune_history,
        'table_costs_history': table_costs_history,
    }, filename)

    filename2 = os.path.join('nbr_channels', f'{args.save_file}.pickle')
    with open(filename2, 'wb') as file:
        pickle.dump(model.num_channels_dict, file)
