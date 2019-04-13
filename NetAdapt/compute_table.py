"""
computes the table from a Network
@Author: Nathan Greffe
take care that this script uses tensorflow and not pytorch, it is also compatible with python 3.5 unlike the others
that only works with python3.6 (f-strings <3 <3)

the compute table generated contains, for each compute_table_on_layer, an (nbr_ch_in * nbr_ch_out) numpy array such
that array[in_ch-1, out_ch-1] = cost of a layer with in_ch and out_ch channels

This script is just a proxy for all the ones defined in utils/compute_table
"""

import pickle
import argparse
import os
from utils.compute_table.wideresnet_table import compute_perf_table_wrn, compute_perf_table_wrn_2_times

parser = argparse.ArgumentParser(description='Computing table')
parser.add_argument('--save_file', default='saveto', type=str, help='file in which to save the table')
parser.add_argument('--net', choices=['res'], default='res')
parser.add_argument('--depth', default=40, type=int, help='depth of net')
parser.add_argument('--width', default=2.0, type=float, help='widen_factor of wideresnet')
parser.add_argument('--num_measures', default=11, type=int, help='number of measures to take teh median from')
parser.add_argument('--img_size', default=32, type=int, help='width and height of the input image')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes we are classifying between')
parser.add_argument('--num_images', default=1, type=int, help='number of images the model makes predictions on at the '
                                                              'same time')
parser.add_argument('--eval_method', choices=['pytorch', 'tf', 'tf-python', 'tf-lite', 'tf-lite-python',
                                              'tf-lite-2-times'],
                    default='tf-python', help='method used to evaluate the model')

# only used with tf-lite, tf and tf-lite-python
parser.add_argument('--tmp_folder', default='/dev/shm/tmp_models', type=str,
                    help='folder in which to create the tmp files, by default, uses linux tmpfs file system')
parser.add_argument('--benchmark_lite_loc', default='/home/pi/tf-lite/benchmark_model', type=str,
                    help='path toward the tf-lite benchmark_model binary')
parser.add_argument('--benchmark_tf_loc', default='/home/pi/tensorflow/benchmark_model', type=str,
                    help='path toward the tf-lite benchmark_model binary')

# only used with tf-lite-2-times
parser.add_argument('--output_folder', default='/media/pi/Elements/models', type=str,
                    help="path towards the .tflite models, either from where we will load them or from where we will "
                         "have to write them")
parser.add_argument('--mode', choices=["load", "save"], type=str, default="load",
                    help="wether we load the .tflite models to benchmark them or save them to be benchmarked on "
                         "another device")
parser.add_argument('--num_process', default=1, type=int,
                    help='only used in save mode, if > 1, means there are several processes computing the'
                         ' same table at the same time and this script thus only has to do a fraction of the work')
parser.add_argument('--offset_process', default=0, type=int,
                    help='only used in save mode,if num_process != 0, is used to determine which fraction of the total '
                         'job this process is doing, must be an int in [0, num_process[')

args = parser.parse_args()

if __name__ == '__main__':
    # computes the table

    if args.net == 'res':
        perf_table = compute_perf_table_wrn(args) if args.eval_method != 'tf-lite-2-times' else \
            compute_perf_table_wrn_2_times(args)

    else:
        raise ValueError('pick a valid net')

    # file saving
    if not (args.eval_method == 'tf-lite-2-times' and args.mode == "save"):
        with open(os.path.join('perf_tables', '{}.pickle'.format(args.save_file)), 'wb') as file:
            pickle.dump(perf_table, file)
