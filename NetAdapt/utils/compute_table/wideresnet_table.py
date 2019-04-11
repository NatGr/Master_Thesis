"""computes the table associated with a wideresnet Network, this script is quite complex bacause we tested many
different methods to compute the tables"""

import numpy as np
import os
import gc
import time


def compute_perf_table_wrn(args):
    if args.eval_method == "pytorch":
        import torch
        import torch.nn as nn
        device = torch.device("cpu")
        print(device)

    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # forces tf to run on cpu, which is what we want to do here
        from tensorflow.keras.models import Model
        from tensorflow.keras import backend as keras_backend
        from tensorflow.keras.layers import Input

        if keras_backend.image_data_format() != 'channels_last':
            raise ValueError('channels_last data format expected')  # channels_last is said to run faster on cpu

        if args.eval_method == "tf-lite" or args.eval_method == "tf-lite-python":
            if not os.path.exists(args.tmp_folder):
                os.makedirs(args.tmp_folder)
            tmp_keras_file = os.path.join(args.tmp_folder, 'model.h5')
            tmp_tflite_file = os.path.join(args.tmp_folder, 'model.tflite')

            if args.eval_method == "tf-lite":
                from .measure_fcts import get_measure_tf_lite
            else:
                from .measure_fcts import get_median_measure_tf_lite_python

        elif args.eval_method == 'tf':
            if not os.path.exists(args.tmp_folder):
                os.makedirs(args.tmp_folder)
            from .measure_fcts import get_measure_tf

        elif args.eval_method == 'tf-python':
            from .measure_fcts import get_median_measure_tf_python

    if args.eval_method == "pytorch":
        from .wideresnet_pytorch import make_conv_model, make_fc_model
    else:
        from .wideresnet_tf import make_conv_model, make_fc_model

    perf_table = {}
    if args.img_size == 32:
        strides = [1, 1, 2, 2]
    else:
        raise ValueError('unsupported input resolution')
    # same as in wideresnet.py, needs to be copied not to have to use an env. with both pytorch and tf
    n_channels = [16, int(16 * args.width), int(32 * args.width), int(64 * args.width)]
    fm_sizes = [args.img_size // stride for stride in strides]

    compute_table_on = [("Conv_0", fm_sizes[0], 3, n_channels[0], strides[0]),
                        ("FC", fm_sizes[3], n_channels[3], 1, None)]
    for i in range(1, 4):
        compute_table_on.append(("Stride_" + str(i), fm_sizes[i - 1], n_channels[i - 1], n_channels[i], strides[i - 1]))
        # used for Skip_i and Conv_i_0_1
        compute_table_on.append(("No_Stride" + str(i), fm_sizes[i], n_channels[i], n_channels[i], 1))
        # used for Conv_i_j_1 and Conv_i_0_2

    for i, (name, width, max_in_channels, max_out_channels, stride) in enumerate(compute_table_on):
        table_entry = np.zeros((max_in_channels, max_out_channels))
        print(str(i) + " table out of " + str(len(compute_table_on)) + " done")

        for in_channels in range(1, max_in_channels + 1):
            print(str(in_channels) + " input_channels out of " + str(max_in_channels))
            for out_channels in range(1, max_out_channels + 1):
                if args.eval_method == "pytorch":
                    measures = np.zeros(args.num_measures)

                    if name == "FC":
                        model = make_fc_model(in_channels, args.num_classes, width, device)
                    else:
                        model = make_conv_model(in_channels, out_channels, stride, device)
                    begin = time.perf_counter()
                    for k in range(args.num_measures):
                        input_tensor = torch.rand(args.num_images, in_channels, width, width, device=device)
                        model(input_tensor)
                        measures[k] = time.perf_counter() - begin
                        table_entry[in_channels - 1, out_channels - 1] = np.median(measures)

                else:
                    inputs = Input(shape=(width, width, in_channels))
                    if name == "FC":
                        model = make_fc_model(inputs, args.num_classes, width)
                    else:
                        model = make_conv_model(inputs, out_channels, stride)

                    if args.eval_method == "tf-python":
                        table_entry[in_channels - 1, out_channels - 1] = get_median_measure_tf_python(
                            model, width, in_channels, number_of_measures=args.num_measures)

                    elif args.eval_method == "tf-lite-python":
                        table_entry[in_channels - 1, out_channels - 1] = get_median_measure_tf_lite_python(
                            model, number_of_measures=args.num_measures, tmp_keras_file=tmp_keras_file,
                            tmp_tflite_file=tmp_tflite_file)

                    elif args.eval_method == "tf-lite":
                        table_entry[in_channels - 1, out_channels - 1] = get_measure_tf_lite(
                            model, number_of_measures=args.num_measures, tmp_keras_file=tmp_keras_file,
                            tmp_tflite_file=tmp_tflite_file, benchmark_loc=args.benchmark_lite_loc)

                    elif args.eval_method == "tf":
                        table_entry[in_channels - 1, out_channels - 1] = get_measure_tf(
                            model, width, in_channels, number_of_measures=args.num_measures,
                            tmp_dir_name=args.tmp_folder, benchmark_loc=args.benchmark_tf_loc)

                    del model
                    keras_backend.clear_session()
                    gc.collect()
        perf_table[name] = table_entry
    return perf_table
