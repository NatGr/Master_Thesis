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
from copy import deepcopy
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
    with open(os.path.join("nbr_channels", f"{pickle_file_name}.pickle"), 'wb') as file:
        pickle.dump((layers, count_ops, count_params), file)


def plot_score(pruning_score_dict, scratch_score_dict, metric):
    """
    plot score evolution wrt pruning from checkpoint files

    :param pruning_score_dict: dict whose keys are the names of the functions to plot and values are lists of name of
    files laying in the checkpoint folder containing the scores (without the .t7 extension), the score inside of a
    same list are averaged
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

    for name, files in pruning_score_dict.items():
        data_list = []
        error_hist_list = []
        for file in files:
            data = torch.load(os.path.join('checkpoints', f'{file}.t7'))
            data_list.append(data[metric])
            error_hist_list.append(data["error_history"])

        data_array = np.array(data_list)
        data_mean = np.mean(data_array, 0)
        error_hist_array = np.array(error_hist_list)
        error_hist_mean = np.mean(error_hist_array, 0)
        plt.plot(data_mean, error_hist_mean, label=name)

    for name, (file_hist, file_pf) in scratch_score_dict.items():
        score = torch.load(os.path.join('checkpoints', f'{file_hist}.t7'))["error_history"][-1]
        with open(os.path.join('nbr_channels', f"{file_pf}.pickle"), 'rb') as file:
            x_data = pickle.load(file)[x_offset]
        plt.plot(x_data, score, label=name, marker='o')

    ax.legend(loc='upper right')
    ax.grid(True)
    plt.show()


def plot_channels_left(files, model, nbr_prunings):
    """
    plots, for each block, the number of middle channels left and the proportion of channels left w.r.t. the initial
    number of channels
    :param files: a list of checkpoint files, in which case we display their mean
    :param model: the corresponding model, this model is assumed to have been pruned more than nbr_prunings times
    :param nbr_prunings: the number of prunings after which we will compute the number of left channels
    """
    nbr_channels_left = []
    prop_channels_left = []
    for file in files:
        model_tmp = deepcopy(model)  # because we cannot modify 2 times the same model in different passes
        state = torch.load(os.path.join('checkpoints', f'{file}.t7'))
        model_tmp(torch.rand(1, 3, 32, 32))
        pruner = Pruner([])
        pruner._get_masks(model_tmp)
        for channel in state['prune_history'][:nbr_prunings]:
            pruner.fixed_prune(model_tmp, channel, verbose=False)
        layers = pruner.compress(model_tmp)
        nbr_channels_left.append([i for i, _ in layers])
        prop_channels_left.append([i/j for i, j in layers])

    # transform the lists in numpy array to ease averaging
    nbr_channels_left = np.array(nbr_channels_left)
    nbr_channels_left_mean = np.mean(nbr_channels_left, 0)
    nbr_channels_left_err = [nbr_channels_left_mean - np.min(nbr_channels_left, 0),
                             np.max(nbr_channels_left, 0) - nbr_channels_left_mean]

    prop_channels_left = np.array(prop_channels_left)
    prop_channels_left_mean = np.mean(prop_channels_left, 0)
    prop_channels_left_err = [prop_channels_left_mean - np.min(prop_channels_left, 0),
                              np.max(prop_channels_left, 0) - prop_channels_left_mean]

    offset_layer = [i for i in range(1, nbr_channels_left.shape[1] + 1)]
    x_ticks = [i for i in range(1, nbr_channels_left.shape[1] + 1, 2)]

    # plot
    fig, ax = plt.subplots(2, 1)
    ax[0].set_title('Number of middle channels left for each block')
    ax[0].set_xlabel('block offset')
    ax[0].set_ylabel('number of channels left')
    ax[0].set_xticks(x_ticks)
    ax[0].bar(x=offset_layer, height=nbr_channels_left_mean, yerr=nbr_channels_left_err)
    ax[1].set_title('Proportion of middle channels for each block')
    ax[1].set_xlabel('block offset')
    ax[1].set_ylabel('proportion of channels left')
    ax[1].set_xticks(x_ticks)
    ax[1].bar(x=offset_layer, height=prop_channels_left_mean, yerr=prop_channels_left_err)
    plt.tight_layout()  # so as to avoid overlaps
    plt.show()
    # plt.savefig('/home/nathan/Documents/TFE/pytorch-prunes/plots/WRN_l1_1000.eps')


def count_nbr_params_flops_acc(file, model, nbr_prunings):
    """
    counts the number of parameters and flops of the model after having pruned the nbr_prunings first channels that were
    pruned in file, also print the accuracy of the model
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
    pruner.compress(model, verbose=False)
    count_ops, count_params = measure_model(model, 32, 32)
    print(f"{count_ops :.2E} FLOPS")
    print(f"{count_params} params")
    print(f"{state['error_history'][nbr_prunings-1]}% of error")  # if nbr_prunings == 0, will return the last element
    # of the array which is what we want


if __name__ == "__main__":
    # save_nbr_channels_and_params("res-40-2-fischer_1299_prunes", WideResNet(40, 2, mask=1), 1100, "nbr_channels/res-40-2_fisher_1100.pickle")

    # plot nbr channel left for Res and Dense Nets
    # plot_channels_left(["res-40-2-l1_1299_prunes"], WideResNet(40, 2, mask=1), 1000)  # resnet-l1-pruning
    # plot_channels_left(["res-40-2-fischer_1299_prunes", "res-40-2_2_fisher_1299_prunes"],
    #                    WideResNet(40, 2, mask=1), 1000)  # resnet-fisher-pruning
    # plot_channels_left(["res-40-2-retrain-1299-pruned", "res-40-2_2_retrain_1300"],
    #                    WideResNet(40, 2, mask=1), 1000)  # resnet-fisher-prune-scratch-prune
    # plot_channels_left(["dense-100-fisher-2600_2000_prunes", "dense-100_2-fisher-2300_2299_prunes"],
    #                    DenseNet(12, 100, 0.5, 10, True, mask=1), 1500)  # densenet-fisher-pruning

    # fisher vs random pruning
    # pruning_dict = {"random pruning": ["res-40-2-random_1299_prunes"],
    #                 "fisher pruning": ["res-40-2-fischer_1299_prunes"],
    #                 "l1 pruning": ["res-40-2-l1_1299_prunes"]}
    # scratch_dict = {"scratch fisher 900 channels": ("res-40-2-scratch-fisher-900", "res-40-2_fisher_900"),
    #                 "scratch l1 900 channels": ("res-40-2-scratch-l1-900", "res-40-2_l1_900"),
    #                 "scratch random 900 channels": ("res-40-2-scratch-random-900", "res-40-2_random_900")}
    # plot_score(pruning_dict, scratch_dict, "param_history")

    # fisher vs prune-retrain-prune
    # pruning_dict = {"fisher pruning": (["res-40-2-fischer_1299_prunes", "res-40-2_2_fisher_1299_prunes"]),
    #                 "fischer_prune_scratch_repeat": (["res-40-2-retrain-1299-pruned"], "res-40-2_2_retrain_1300"])}
    # scratch_dict = {"scratch fisher 600 channels": ("res-40-2-scratch-fisher-600", "res-40-2_fisher_600"),
    #                 "scratch fisher 900 channels": ("res-40-2-scratch-fisher-900", "res-40-2_fisher_900"),
    #                 "scratch fisher 1100 channels": ("res-40-2-scratch-fisher-1100", "res-40-2_fisher_1100")}
    # plot_score(pruning_dict, scratch_dict, "param_history")

    # not-so-dense net
    # pruning_dict = {"densenet-fisher": ["dense-100-fisher-2300_2299_prunes", "dense-100_2-fisher-2300_2299_prunes"]}
    # scratch_dict = {"notsodense": ("notsodense1-100-k=3", "notsodense1-100-k=3_0")}
    # plot_score(pruning_dict, scratch_dict, "param_history")

    # accuracies:
    # print("------ notsodense1")
    # count_nbr_params_flops_acc("notsodense1-100-k=3", NotSoDenseNet1(12, 100, 0.5, 10, True, 3, mask=1), 0)
    # print("------ densenet")
    # count_nbr_params_flops_acc("dense-100", DenseNet(12, 100, 0.5, 10, True, mask=1), 0)
    # count_nbr_params_flops_acc("dense-100_2", DenseNet(12, 100, 0.5, 10, True, mask=1), 0)
    # print("------ densenet-cosine")
    # count_nbr_params_flops_acc("dense-100-cosine", DenseNet(12, 100, 0.5, 10, True, mask=1), 0)
    # count_nbr_params_flops_acc("dense-100-cosine_2", DenseNet(12, 100, 0.5, 10, True, mask=1), 0)

    # get flops and params for morphnet models
    # file_name = "res-40-2-rs=8e-9-tresh=1e-3-lr=1e-2"
    # with open(f"../morph_net/pickle/{file_name}.pickle", 'rb') as file:
    #     channels_dict = {name: channels_rem for name, (channels_rem, _) in pickle.load(file).items()}
    #     model = WideResNetAllLayersPrunable(40, channels_dict=channels_dict)
    # count_nbr_params_flops_acc(f"morphnet/morphnet-{file_name}", model, 0)

    # get flops and params for fisher models
    file_hist = "res-40-2-scratch-fisher-450"
    file_pf = "res-40-2_fisher_450"
    score = torch.load(os.path.join('checkpoints', f'{file_hist}.t7'))["error_history"][-1]
    with open(os.path.join('nbr_channels', f"{file_pf}.pickle"), 'rb') as file:
        data = pickle.load(file)
    print(f"{data[0]} layers")
    print(f"{data[1] :.2E} FLOPS")
    print(f"{data[2]} params")
    print(f"{score}% of error")
