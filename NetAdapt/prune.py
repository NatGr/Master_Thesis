"""NetAdapt Pruning script
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
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--learning_rate', default=8e-4, type=float, help='initial learning rate')
parser.add_argument('--init_red_fact', default=20, type=int,
                    help='fraction of the initial inference time that will be pruned at the first step, i.e. if network'
                         ' takes .1s to make a prediction, and init_red_fact = 20, the network will have an inference '
                         'time of approximately 19/20 * 0.1s after one pruning step')
parser.add_argument('--decay_rate', default=0.98, type=float, help='rate at which the target resource reduction for '
                                                                   'one step is reduced')
parser.add_argument('--holdout_prop', default=0.1, type=float, help='fraction of training set used for holdout')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float, help='weight decay')
parser.add_argument('--short_term_fine_tune', default=100, type=int, help='number of batches ')
parser.add_argument('--double_fine_tune', action='store_true',
                    help="fine tunes 2 times between each pruning (one to fine tune and one to compute fisher coefs"
                         "the one to compute fisher coefs will be 100 batches long and the first one "
                         "'short_term_fine_tune' batches long)")
parser.add_argument('--long_term_fine_tune', default=0, type=int, help='long term fine tune on the whole dataset, '
                                                                       'set to 0 by default because training from '
                                                                       'scratch gives better results')
parser.add_argument('--pruning_method', choices=['fisher', 'l2'], type=str, default='fisher',
                    help='pruning algo to use')
parser.add_argument('--allow_small_prunings', action='store_true',
                    help="allows to prune from a layer even if it doesn't make us achieve the reduction objective")
parser.add_argument('--width', default=2.0, type=float, metavar='D')
parser.add_argument('--depth', default=40, type=int, metavar='W')


args = parser.parse_args()
print(args)

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
train_loader, holdout_loader = get_train_holdout(args.data_loc, args.workers, args.batch_size, args.holdout_prop)


if __name__ == '__main__':

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    prune_history.append(None)
    table_costs_history.append(model.total_cost)
    error_history.append(validate(model, val_loader, criterion, device, memory_leak=True))

    # first fine-tune to have gradients so that we can fisher-prune some layers
    finetune(model, optimizer, criterion, args.short_term_fine_tune, train_loader, "", device)

    # validate on holdout so that we can compute the error change after having pruned one layer
    prev_holdout_error = validate(model, holdout_loader, criterion, device, val_set_name="holdout", memory_leak=True)

    red_objective = (1 - args.pruning_fact) * model.total_cost
    target_gains = args.pruning_fact * model.total_cost / args.init_red_fact  # gains at first epoch to achieve the
    # objective
    step_number = 1

    while model.total_cost > red_objective:

        print(f"Pruning step number {step_number} -- target_gains are {target_gains}s:")

        # Prune
        best_network, best_error, best_gains, pruned_layer, number_pruned = None, None, None, None, None

        # done in two steps to reduce number of memory transfers
        layer_mask_channels_gains = []
        for layer in model.to_prune:
            num_channels, gains = model.choose_num_channels(layer, target_gains, args.allow_small_prunings)
            if num_channels is not None:
                remaining_channels = model.choose_which_channels(layer, num_channels,
                                                                 100 if args.double_fine_tune else
                                                                 args.short_term_fine_tune,
                                                                 use_fisher=args.pruning_method == 'fisher')
                layer_mask_channels_gains.append((layer, remaining_channels, num_channels, gains))

        model.reset_fisher()  # cleans run_fish from memory
        model.cpu()  # stores it on CPU to avoid having 2 models on GPU at the same time

        for layer, remaining_channels, new_num_channels_pruned, new_gains in layer_mask_channels_gains:

            # torch.cuda.empty_cache()
            # print(torch.cuda.memory_allocated() / 10 ** 9, 'GB allocated')  # in case there are memory leaks

            # creates a new model with the new mask to be fine_tuned
            new_model = build_model(args, device, model)
            new_model.load_bigger_state_dict(model.state_dict())  # copy weights and stuff
            new_model.perf_table = model.perf_table
            new_model.total_cost = model.total_cost - new_gains
            new_model.num_channels_dict = model.num_channels_dict.copy()
            new_model.prune_channels(layer, remaining_channels)

            new_model.to(device)

            optimizer = torch.optim.SGD([v for v in new_model.parameters() if v.requires_grad],
                                        lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            finetune(new_model, optimizer, criterion, args.short_term_fine_tune, train_loader, layer, device)
            if args.double_fine_tune:
                finetune(new_model, optimizer, criterion, 100, train_loader, layer, device)

            new_error = validate(new_model, holdout_loader, criterion, device, val_set_name="holdout", memory_leak=True)

            delta_error = new_error - prev_holdout_error
            relative_delta_error = delta_error / new_gains

            print(f"layer {layer} \t channels pruned {new_num_channels_pruned} \t error increase {delta_error :.2f} \t "
                  f"predicted gains {new_gains :.4f} \t ratio {relative_delta_error :.2f}\n")

            prev_delta = best_error - prev_holdout_error if best_error is not None else 0

            # if we lose precision, best error_increase/cost_decrease ratio wins, otherwise,
            # better gain of precision wins
            if best_error is None or (prev_delta >= 0 and relative_delta_error < prev_delta / best_gains) \
                                  or (prev_delta < 0 and delta_error < prev_delta):
                new_model.cpu()
                best_network, best_error, pruned_layer = new_model, new_error, layer
                best_gains, number_pruned = new_gains, new_num_channels_pruned
            else:
                del new_model

        if best_network is None:
            raise Exception('We could not find a single layer to prune')
        print(f"the best validation error achieved was of {best_error} for layer {pruned_layer};"
              f" {number_pruned} channels were pruned; inference time gains of {best_gains :.4f}s")
        torch.cuda.empty_cache()  # frees GPU memory to avoid running out of ram
        best_network.to(device)
        model = best_network
        prev_holdout_error = best_error

        prune_history.append((pruned_layer, number_pruned))
        table_costs_history.append(model.total_cost)

        # evaluate on validation set
        error_history.append(validate(model, val_loader, criterion, device, memory_leak=True))

        # prepare next step
        step_number += 1
        target_gains *= args.decay_rate

    print(f"pruned network inference time according to perf_table: {model.total_cost :.4f}")

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
        'epoch': step_number,
        'state_dict': model.state_dict(),
        'error_history': error_history,
        'prune_history': prune_history,
        'table_costs_history': table_costs_history,
    }, filename)

    filename2 = os.path.join('nbr_channels', f'{args.save_file}.pickle')
    with open(filename2, 'wb') as file:
        pickle.dump(model.num_channels_dict, file)
