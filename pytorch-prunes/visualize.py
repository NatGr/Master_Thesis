"""
this script contains functions to:
    - make a plot to compare the performance of the different NN
    - print the number of channels that survived pruning
Author: Nathan Greffe
"""
import matplotlib.pyplot as plt
import torch
import os
from models import *
from funcs import *


def plot_score(score_dict, metric):
    """
    plot score evolution wrt pruning from checkpoint files

    :param score_dict: dict whose keys are the names of the functions to plot and values are the name of the file
    laying in the checkpoint folder containing the scores (without the .t7 extension)
    :param metric: the name of the metric to use on the x-axis (must be a key of the checkpoint file)
    """
    fig, ax = plt.subplots()
    ax.set_title('Evolution of the test error', fontsize=15, fontweight='bold')
    ax.set_xlabel(metric)
    ax.set_ylabel('test error')

    for name, file in score_dict.items():
        data = torch.load(os.path.join('checkpoints', f'{file}.t7'))
        plt.plot(data[metric], data["error_history"], label=name)

    ax.legend(loc='upper right')
    plt.show()


def plot_channels_left(file, model, nbr_prunings):
    """
    plots, for each block, the number of channels left and the proportion of channels left w.r.t. the initial number
    of channels
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
    channels = pruner.compress(model)


if __name__ == "__main__":
    plot_channels_left("res-40-2-retrain-1299-pruned", WideResNet(40, 2, mask=1), 500)

    # plot_score({"random pruning": "res-40-2-random_1299_prunes",
    #             "fisher pruning": "res-40-2-fischer_1299_prunes",
    #             "fischer_prune_scratch_repeat": "res-40-2-retrain-1299-pruned"},
    #            "param_history")

