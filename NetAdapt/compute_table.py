"""
computes the table from a Network
@Author: Nathan Greffe
take care that this script uses tensorflow and not pytorch, it is also compatible with python 2.7 unlike all the others
that only works with python3

the compute table generated contains, for each compute_table_on_layer, an (nbr_ch_in * nbr_ch_out) numpy array such
that array[in_ch-1, out_ch-1] = cost of a layer with in_ch and out_ch channels
"""
from __future__ import print_function

import time
import pickle
import numpy as np
import argparse
import os
import subprocess

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
parser.add_argument('--eval_method', choices=['pytorch', 'tf', 'tf-lite', 'tf-lite-python'], default='tf',
                    help='method used to evaluate the model')

# only used with tf-lite and tf-lite-python
parser.add_argument('--tmp_folder', default='/dev/shm/tmp_models', type=str,
                    help='folder in which to create the tmp files, by default, uses linux tmpfs file system')
parser.add_argument('--benchmark_loc', default='/home/pi/tf-lite/benchmark_model', type=str,
                    help='path toward the tf-lite benchmark_model binary')

args = parser.parse_args()
if args.eval_method == "pytorch":
    import torch
    import torch.nn as nn
    device = torch.device("cpu")
    print(device)

else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # forces tf to run on cpu, which is what we want to do here
    from tensorflow.keras.layers import BatchNormalization, Conv2D, AveragePooling2D, Dense, Input, Activation, Flatten
    from tensorflow.keras.models import Model, save_model
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras import backend as keras_backend
    from tensorflow import lite

    if keras_backend.image_data_format() != 'channels_last':
        raise ValueError('channels_last data format expected')  # channels_last is said to run faster on cpu

    if args.eval_method == "tf-lite" or args.eval_method == "tf-lite-cpu":
        if not os.path.exists(args.tmp_folder):
            os.makedirs(args.tmp_folder)
        tmp_keras_file = os.path.join('tmp', 'model.h5')
        tmp_tflite_file = os.path.join('tmp', 'model.tflite')


        def get_measure_tf_lite(model, number_of_measures=args.num_measures, tmp_keras_file=tmp_keras_file,
                                tmp_tflite_file=tmp_tflite_file, benchmark_loc=args.benchmark_loc):
            """given a model, loads that model in tf_lite and benchmarks the time needed for a prediction in C++ using
            the benchmark too associated with tf-lite (this tool does not return median but only mean so we will use
            that instead)
            :return: the mean of number_of_measures trials"""
            model.compile(optimizer=SGD(), loss='binary_crossentropy')
            save_model(model, tmp_keras_file)

            # Convert to TensorFlow Lite model.
            converter = lite.TFLiteConverter.from_keras_model_file(tmp_keras_file)
            tflite_model = converter.convert()
            with open(tmp_tflite_file, "wb") as file:
                file.write(tflite_model)

            # Loads TFLite model and get measures
            command_line = benchmark_loc + " --graph=" + tmp_tflite_file + \
                           " --min_secs=0 --warmup_min_secs=0 --num_runs=" + number_of_measures + \
                           " |& tr -d '\n' | awk '{print $NF}'"  # tr removes the \n and awk gets the last element of
            # the outputs message, |& is used before tr because we want to pipe strderr and not stdout
            result = int(subprocess.check_output(command_line, shell=True)) / 10**6  # result given in microseconds
            return result


        def get_median_measure_tf_lite_python(model, number_of_measures=args.num_measures,
                                              tmp_keras_file=tmp_keras_file, tmp_tflite_file=tmp_tflite_file):
            """given a model, loads that model in tf_lite and benchmarks the time needed for a prediction in python
            :return: the median of number_of_measures trials"""
            measures = np.zeros(number_of_measures)

            model.compile(optimizer=SGD(), loss='binary_crossentropy')
            save_model(model, tmp_keras_file)

            # Convert to TensorFlow Lite model.
            converter = lite.TFLiteConverter.from_keras_model_file(tmp_keras_file)
            tflite_model = converter.convert()
            with open(tmp_tflite_file, "wb") as file:
                file.write(tflite_model)

            # Load TFLite model and get measures
            interpreter = lite.Interpreter(model_path=tmp_tflite_file)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()

            for k in range(number_of_measures):
                # Test model on random input data.
                input_shape = input_details[0]['shape']
                input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)

                begin = time.perf_counter()
                interpreter.invoke()
                measures[k] = time.perf_counter() - begin

            return np.median(measures)

    else:
        def get_median_measure_tf(model, number_of_measures=args.num_measures):
            """given a model, get the median measure without using tflite
            :return: the median of number_of_measures trials"""
            measures = np.zeros(number_of_measures)

            model.compile(optimizer=SGD(), loss='binary_crossentropy')

            for k in range(number_of_measures):
                # Test model on random input data.
                input_data = np.array(np.random.random_sample((10, width, width, in_channels)), dtype=np.float32)

                begin = time.time()  # time.time() to be compatible with python 2.7
                model.predict(input_data, batch_size=10)
                measures[k] = time.time() - begin

            return np.median(measures)


# implements the functions needed to build a model
if args.net == 'res':
    if args.eval_method == "pytorch":
        def make_conv_model(in_channels, out_channels, stride, device):
            """creates a small sequential model composed of a convolution, a batchnorm and a relu activation
            the model is set to eval mode since it is used to measure evaluation time"""
            model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            model.to(device)
            model.eval()
            return model


        def make_fc_model(in_channels, num_classes, width, device):
            """creates a small sequential model composed of an average pooling and a fully connected layer
            the model is set to eval mode since it is used to measure evaluation time"""

            class Flatten(nn.Module):  # not defined in pytorch
                def forward(self, x):
                    return x.view(x.size(0), -1)

            model = nn.Sequential(
                nn.AvgPool2d(width),
                Flatten(),
                nn.Linear(in_channels, num_classes)
            )
            model.to(device)
            model.eval()
            return model

    else:
        def make_conv_model(inputs, out_channels, stride):
            """creates a small sequential model composed of a convolution, a batchnorm and a relu activation"""
            outputs = Conv2D(out_channels, kernel_size=3, strides=stride, padding="same", use_bias=False)(inputs)
            outputs = BatchNormalization()(outputs)
            outputs = Activation('relu')(outputs)
            return Model(inputs=inputs, outputs=outputs)

        def make_fc_model(inputs, num_classes, width):
            """creates a small sequential model composed of an average pooling and a fully connected layer"""
            outputs = AveragePooling2D(pool_size=width)(inputs)
            outputs = Flatten()(outputs)
            outputs = Dense(units=num_classes)(outputs)
            return Model(inputs=inputs, outputs=outputs)

else:
    raise ValueError('pick a valid net')


if __name__ == '__main__':
    # computes the table
    perf_table = {}

    if args.net == 'res':
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
            compute_table_on.append(("Stride_" + str(i), fm_sizes[i-1], n_channels[i-1], n_channels[i], strides[i-1]))
            # used for Skip_i and Conv_i_0_1
            compute_table_on.append(("No_Stride" + str(i), fm_sizes[i], n_channels[i], n_channels[i], 1))
            # used for Conv_i_j_1 and Conv_i_0_2

        for i, (name, width, max_in_channels, max_out_channels, stride) in enumerate([compute_table_on[2]]):
            table_entry = np.zeros((max_in_channels, max_out_channels))
            print(str(i) + "table out of" + str(len(compute_table_on)) + "done")

            for in_channels in range(1, max_in_channels + 1):
                print(str(in_channels) + "input_channels out of" + max_in_channels)
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

                        if args.eval_method == "tf":
                            table_entry[in_channels - 1, out_channels - 1] = get_median_measure_tf(model)
                        elif args.eval_method == "tf-lite-python":
                            table_entry[in_channels - 1, out_channels - 1] = get_median_measure_tf_lite(model)
                        else:
                            table_entry[in_channels - 1, out_channels - 1] = get_measure_tf_lite(model)
                        del model
                        keras_backend.clear_session()
            perf_table[name] = table_entry

    else:
        raise ValueError('pick a valid net')

    # file saving
    with open(os.path.join('perf_tables', str(args.save_file) + '.pickle'), 'wb') as file:
        pickle.dump(perf_table, file)
