"""file containing the functions used to measure the performance of a model using tensorflow"""

import time
import os
import subprocess
import numpy as np
from tensorflow.keras import backend as keras_backend
from tensorflow import lite
import tensorflow as tf
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.optimizers import SGD


def get_measure_tf_lite(model, number_of_measures, tmp_keras_file, tmp_tflite_file, benchmark_loc):
    """given a model, loads that model in tf_lite and benchmarks the time needed for a prediction in C++ using
    the benchmark tool associated with tf-lite (this tool does not return median but only mean so we will use
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
    command_line = "{} --graph={} --min_secs=0 --warmup_min_secs=0 --num_runs={} |& tr -d '\n' | awk {}".format(
        benchmark_loc, tmp_tflite_file, number_of_measures, "'{print $NF}'")  # tr removes the \n and awk gets
    # the last element of the outputs message, |& is used before tr because we want to pipe strderr and not
    # stdout
    result = float(subprocess.check_output(command_line, shell=True, executable='/bin/bash')) / 10 ** 6  # result
    # given in microseconds
    return result


def get_median_measure_tf_lite_python(model, number_of_measures, tmp_keras_file, tmp_tflite_file):
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


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    taken from: https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def get_measure_tf(model, in_width, in_channels, number_of_measures, tmp_dir_name, benchmark_loc,
                   tmp_file_name='model.pb'):
    """given a model, saves that model as a .pb file and benchmarks the time needed for a prediction in C++
    using the benchmark tool associated with tf (this tool does not return median but only mean so we will use
    that instead)
    :return: the mean of number_of_measures trials
    As of now, this function does not work with my benchmark binary since the binary wasn't compiled at the
    same time than the whl pckg I used on the rasberry pi (I couldn't make the cross compilation work) so I got
    a whl for tf 1.13.1 for rasberry-pi on the internet"""
    model.compile(optimizer=SGD(), loss='binary_crossentropy')

    frozen_graph = freeze_session(keras_backend.get_session(),
                                  output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, tmp_dir_name, tmp_file_name, as_text=False)

    # Loads model and get measures
    command_line = "{} --graph={} --input_layer_shape='1, {}, {}, {}' --num_runs={} |& tr -d '\n' | awk {}" \
        .format(benchmark_loc, os.path.join(tmp_dir_name, tmp_file_name), in_width,
                in_width, in_channels, number_of_measures, "'{print $NF}'")

    result = subprocess.check_output(command_line, shell=True, executable='/bin/bash')
    result = float(result) / 10 ** 6
    # result given in microseconds
    return result


def get_median_measure_tf_python(model, in_width, in_channels, number_of_measures):
    """given a model, get the median measure without using tflite
    :return: the median of number_of_measures trials"""
    measures = np.zeros(number_of_measures)

    model.compile(optimizer=SGD(), loss='binary_crossentropy')

    for k in range(number_of_measures):
        # Test model on random input data.
        input_data = np.array(np.random.random_sample((1, in_width, in_width, in_channels)), dtype=np.float32)

        begin = time.perf_counter()
        model.predict(input_data, batch_size=1)
        measures[k] = time.perf_counter() - begin

    return np.median(measures)
