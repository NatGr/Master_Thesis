"""Pruning script
@Author: Nathan Greffe
"""

import argparse

from models.wideresnet import *
from utils.training_fcts import *
from utils.data_fcts import *

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
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--learning_rate', default=8e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--steps', default=20, type=int, metavar='epochs', help='no. pruning steps')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float, metavar='W', help='weight decay')
parser.add_argument('--short_term_fine_tune', default=100, type=int, help='number of batches ')
parser.add_argument('--long_term_fine_tune', default=0, type=int, help='long term fine tune on the whole dataset, '
                                                                       'set to 0 by default because training from '
                                                                       'scratch gives better results')
parser.add_argument('--pruning_method', choices=['fisher', 'weight_l2'], type=str, default='fisher',
                    help='pruning algo to use')
parser.add_argument('--width', default=2.0, type=float, metavar='D')
parser.add_argument('--depth', default=40, type=int, metavar='W')


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def build_model(args, device, prev_model=None):
    """build the model given the arguments on the device, take care that you still need to call model.to(device)
    if prev_model is not None, we will copy the number of channels in all the layers of prev_model"""
    if args.net == 'res':
        return WideResNet(args.depth, args.width, device, prev_model)
    else:
        raise ValueError('pick a valid net')


model = build_model(args, device)
error_history = []
prune_history = []
table_costs_history = []
model.load_state_dict(torch.load(os.path.join('checkpoints', f'{args.base_model}.t7'),
                                 map_location='cpu')['state_dict'], strict=True)
model.to(device)
model.load_table(os.path.join("perf_tables", f"{args.perf_table}.pickle"))

print('==> Preparing data..')

# base datasets
full_train_loader, val_loader = get_full_train_val(args.data_loc, args.workers, args.batch_size)

# dataset split into train/holdout set
train_loader, holdout_loader = get_train_holdout(args.data_loc, args.workers, args.batch_size)


if __name__ == '__main__':

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    prune_history.append(None)
    table_costs_history.append(model.total_cost)
    error_history.append(validate(model, val_loader, criterion, device, memory_leak=True))

    finetune(model, optimizer, criterion, args.short_term_fine_tune, train_loader, "", device)  # first fine-tune to
    # have gradients so that we can fisher-prune some layers

    target_gains = args.pruning_fact * model.total_cost / args.steps  # gains at each epoch to achieve the reduction

    for epoch in range(1, args.steps+1):

        print(f"Epoch {epoch} -- lr is {optimizer.param_groups[0]['lr']}:")

        # Prune
        best_network, best_error, best_gains, pruned_layer, number_pruned = None, None, None, None, None

        # done in two steps to reduce number of memory transfers
        layer_mask_channels_gains = []
        for layer in model.to_prune:
            num_channels, gains = model.choose_num_channels(layer, target_gains)
            if num_channels is not None:
                remaining_channels = model.choose_which_channels(layer, num_channels, args.short_term_fine_tune,
                                                                 use_fisher=args.pruning_method == 'fisher')
                layer_mask_channels_gains.append((layer, remaining_channels, num_channels, gains))

        model.reset_fisher()  # cleans run_fish from memory
        model.cpu()  # stores it on CPU to avoid having 2 models on GPU at the same time

        for layer, remaining_channels, new_num_channels, new_gains in layer_mask_channels_gains:

            prev_error = error_history[-1]

            # torch.cuda.empty_cache()
            # print(torch.cuda.memory_allocated() / 10 ** 9, 'GB allocated')  # in case there are memory leaks

            # creates a new model with the new mask to be fine_tuned
            new_model = build_model(args, device, model)
            new_model.load_bigger_state_dict(model.state_dict())  # copy weights and stuff
            new_model.perf_table = model.perf_table
            new_model.total_cost = model.total_cost - new_gains
            new_model.num_channels_dict = model.num_channels_dict
            new_model.prune_channels(layer, remaining_channels)

            new_model.to(device)

            optimizer = torch.optim.SGD([v for v in new_model.parameters() if v.requires_grad],
                                        lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            finetune(new_model, optimizer, criterion, args.short_term_fine_tune, train_loader, layer, device)

            new_error = validate(new_model, holdout_loader, criterion, device, val_set_name="holdout", memory_leak=True)

            if best_error is None or (new_error - prev_error)/new_gains < (best_error - prev_error)/best_gains:
                new_model.cpu()
                best_network, best_error, pruned_layer, number_pruned = new_model, new_error, layer, new_num_channels
                best_gains = new_gains
            else:
                del new_model

        if best_network is None:
            raise Exception('We could not find a single layer to prune')
        print(f"the best validation error achieved was of {best_error} for layer {pruned_layer};"
              f" {number_pruned} channels were pruned; inference time gains of {gains}s")
        torch.cuda.empty_cache()  # frees GPU memory to avoid running out of ram
        best_network.to(device)
        model = best_network

        prune_history.append((pruned_layer, number_pruned))
        table_costs_history.append(model.total_cost)

        # evaluate on validation set
        error_history.append(validate(model, val_loader, criterion, device, memory_leak=True))

    # long term fine tune
    if args.long_term_fine_tune != 0:
        optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                    lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        finetune(model, optimizer, criterion, args.long_term_fine_tune, full_train_loader, "", device)
        error_history.append(validate(model, val_loader, criterion, device, memory_leak=True))
        prune_history.append(None)
        table_costs_history.append(table_costs_history[-1])

    # Save
    filename = os.path.join('checkpoints', f'{args.save_file}.t7')
    torch.save({
        'epoch': args.steps,
        'state_dict': model.state_dict(),
        'error_history': error_history,
        'prune_history': prune_history,
        'table_costs_history': table_costs_history,
    }, filename)

    filename2 = os.path.join('nbr_channels', f'{args.save_file}.pickle')
    with open(filename2, 'wb') as file:
        pickle.dump(model.num_channels_dict, file)
