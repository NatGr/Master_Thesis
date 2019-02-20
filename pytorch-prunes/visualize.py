"""
this script contains functions to:
    - make a plot to compare the performance of the different NN
    - print the number of channels that survived pruning
Author: Nathan Greffe
"""
import matplotlib.pyplot as plt
import torch
import os
import seaborn as sns
from models import *
from funcs import *
import pickle


def save_nbr_channels_and_params(file, model, nbr_prunings, pickle_file_name):
    """
    saves a tuple (as a pickle file) whose first elements is a list containing the number of channels per layer,
    second element is the number of parameters in the network and third element the number of FLOPS
    :param file: the checkpoint file of the NN
    :param model: the corresponding model, this model is assumed to have been pruned more than nbr_prunings times
    :param nbr_prunings: the number of prunings after which we will compute the number of left channels
    :param pickle_file_name: the name of the pickle file that will contain the data
    """
    state = torch.load(os.path.join('checkpoints', f'{file}.t7'))
    model(torch.rand(1, 3, 32, 32))
    pruner = Pruner([])
    if nbr_prunings != 0:  # so that it works with model trained without pruning as well
        pruner._get_masks(model)
        for channel in state['prune_history'][:nbr_prunings]:
            pruner.fixed_prune(model, channel, verbose=False)
    layers = pruner.compress(model)
    layers = [i for i, _ in layers]  # get rid of initial number of layers
    count_ops, count_params = measure_model(model, 32, 32)
    with open(pickle_file_name, 'wb') as file:
        pickle.dump((layers, count_ops, count_params), file)


def plot_score(pruning_score_dict, scratch_score_dict, metric):
    """
    plot score evolution wrt pruning from checkpoint files

    :param pruning_score_dict: dict whose keys are the names of the functions to plot and values are the name of the
    file laying in the checkpoint folder containing the scores (without the .t7 extension)
    :param scratch_score_dict:  dict whose keys are the names of the points to plot and values are the names of 2 files,
    one containing the checkpoint of the trained model from scratch and one the number of parameters
    :param metric: the name of the metric to use on the x-axis (must be a key of the checkpoint file)
    """
    fig, ax = plt.subplots()
    if metric == "param_history":
        x_offset = 2  # used in scratch_score_dict
        x_label = "nbr of parameters"
    else:
        x_offset = 1
        x_label = "nbr of FLOPS"

    ax.set_title('Evolution of the test error', fontsize=15, fontweight='bold')
    ax.set_xlabel(x_label)
    ax.set_ylabel('test error')

    for name, file in pruning_score_dict.items():
        data = torch.load(os.path.join('checkpoints', f'{file}.t7'))
        plt.plot(data[metric], data["error_history"], label=name)

    for name, (file_hist, file_pf) in scratch_score_dict.items():
        score = torch.load(os.path.join('checkpoints', f'{file_hist}.t7'))["error_history"][-1]
        with open(os.path.join('nbr_channels', f"{file_pf}.pickle"), 'rb') as file:
            x_data = pickle.load(file)[x_offset]
        plt.plot(x_data, score, label=name, marker='o')

    ax.legend(loc='upper right')
    ax.grid(True)
    plt.show()


def plot_channels_left(file, model, nbr_prunings):
    """
    plots, for each block, the number of middle channels left and the proportion of channels left w.r.t. the initial
    number of channels
    :param file: the checkpoint file of the NN
    :param model: the corresponding model, this model is assumed to have been pruned more than nbr_prunings times
    :param nbr_prunings: the number of prunings after which we will compute the number of left channels
    """
    state = torch.load(os.path.join('checkpoints', f'{file}.t7'))
    model(torch.rand(1, 3, 32, 32))
    pruner = Pruner([])
    pruner._get_masks(model)
    for channel in state['prune_history'][:nbr_prunings]:
        pruner.fixed_prune(model, channel, verbose=False)
    layers = pruner.compress(model)
    offset_layer = [i for i in range(1, len(layers) + 1)]
    nbr_channels_left = [i for i, _ in layers]
    prop_channels_left = [i/j for i, j in layers]

    fig, ax = plt.subplots(2, 1)
    ax[0].set_title('Number of middle channels left for each block')
    ax[0].set_xlabel('block offset')
    ax[0].set_ylabel('number of channels left')
    sns.barplot(x=offset_layer, y=nbr_channels_left, ax=ax[0], color='steelblue')
    ax[1].set_title('Proportion of middle channels for each block')
    ax[1].set_xlabel('block offset')
    ax[1].set_ylabel('proportion of channels left')
    sns.barplot(x=offset_layer, y=prop_channels_left, ax=ax[1], color='steelblue')
    plt.tight_layout()  # so as to avoid overlaps
    plt.show()
    # plt.savefig('/home/nathan/Documents/TFE/pytorch-prunes/plots/WRN_l1_1000.eps')


def count_nbr_params_and_flops(file, model, nbr_prunings):
    """
    counts the number of parameters and flops of the model after having pruned the nbr_prunings first channels that were
    pruned in file
    :param file: the checkpoint file of the NN
    :param model: the corresponding model, this model is assumed to have been pruned more than nbr_prunings times
    :param nbr_prunings: the number of prunings after which we will compute the number of params and flops
    """
    state = torch.load(os.path.join('checkpoints', f'{file}.t7'))
    model(torch.rand(1, 3, 32, 32))
    pruner = Pruner([])
    if nbr_prunings != 0:
        pruner._get_masks(model)
        for channel in state['prune_history'][:nbr_prunings]:
            pruner.fixed_prune(model, channel, verbose=False)
    pruner.compress(model)
    count_ops, count_params = measure_model(model, 32, 32)
    print(f"{count_ops :.2E} FLOPS")
    print(f"{count_params} params")


if __name__ == "__main__":
    save_nbr_channels_and_params("res-40-2-fischer_1299_prunes", WideResNet(40, 2, mask=1), 1100, "nbr_channels/res-40-2_fisher_1100.pickle")

    # plot_channels_left("res-40-2-l1_1299_prunes", WideResNet(40, 2, mask=1), 1000)

    pruning_dict = {"random pruning": "res-40-2-random_1299_prunes",
                    "fisher pruning": "res-40-2-fischer_1299_prunes",
                    "fischer_prune_scratch_repeat": "res-40-2-retrain-1299-pruned",
                    "l1 pruning": "res-40-2-l1_1299_prunes",
                    "densenet-fisher": "dense-100-fisher-2600_2000_prunes"}
    scratch_dict = {"scratch fisher 900 channels": ("res-40-2-scratch-fisher-900", "res-40-2_fisher_900"),
                    "scratch l1 900 channels": ("res-40-2-scratch-l1-900", "res-40-2_l1_900"),
                    "scratch random 900 channels": ("res-40-2-scratch-random-900", "res-40-2_random_900"),
                    "notsodense": ("notsodense1-100-k=3", "notsodense1-100-k=3_0")}
    # plot_score(pruning_dict, scratch_dict, "param_history")

