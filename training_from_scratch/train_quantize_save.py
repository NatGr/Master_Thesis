"""trains a network for a given number of epochs with quantization aware training and then saves
it as a tf-lite model"""
import argparse
import json
import os
import tensorflow as tf
import subprocess
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.regularizers import l2
from LRTensorBoard import LRTensorBoard
from KDDataGenerator import KDDataGenerator

parser = argparse.ArgumentParser(description='Training and saving in tf_lite')
parser.add_argument('--workers', default=1, type=int, help='number of data loading workers')
parser.add_argument('--save_file', default='saveto', type=str, help='name to use for this file')
parser.add_argument('--tmp_folder', default='tmp_models', type=str,
                    help='folder in which to create the tmp files (for tf-lite conversion)')
parser.add_argument('--train_val_set', action='store_true',
                    help='uses 10% of training set as validation set to tune hyperparameters; setting this argument'
                         'overrides "no_tf_lite_conversion" to True and adds "_val" on the file name')
parser.add_argument('--get_tf_lite', action='store_true',
                    help='gets a tf_lite_model corresponding to what was trained')

# Learning specific arguments
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--learning_rate', default=.1, type=float, help='initial learning rate')
parser.add_argument('--lr_type', default='multistep', type=str, help='learning rate strategy (default: multistep)',
                    choices=['multistep'])
parser.add_argument('--epochs', default=250, type=int, help='no. epochs')
parser.add_argument('--epoch_step', default='[60,120,160]', type=str, help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay factor')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=0.0001, type=float, help='weight decay')

# Net specific arguments
parser.add_argument('--depth', '-d', default=40, type=int, help='depth of network')
parser.add_argument('--width', default=32, type=int,
                    help='width of the first subnetwork, width are multiplied by 2 at each application of stride, ')

args = parser.parse_args()
print(args)

LOGS_DIR = 'tf_logs'
MODELS_DIR = 'tf_lite_models'
TEACHER_DIR = 'teacher_pred'
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
if not os.path.exists(TEACHER_DIR):
    os.makedirs(TEACHER_DIR)
if not os.path.exists(args.tmp_folder):
    os.makedirs(args.tmp_folder)


def build_keras_model(args):
    blocks_per_subnet = [int(args.depth / 3)] * 3
    regularizer = l2(args.weight_decay)
    channels_per_subnet = [args.width, args.width * 2, args.width * 4]

    layers = [tf.keras.layers.Conv2D(16, kernel_size=3, activation="relu", padding="same", use_bias=False,
                                     kernel_regularizer=regularizer, input_shape=(32, 32, 3))]

    for i in range(3):
        for j in range(blocks_per_subnet[i]):
            layers.append(tf.keras.layers.DepthwiseConv2D(kernel_size=3, activation="relu",
                                                          strides=2 if i != 0 and j == 0 else 1, padding="same",
                                                          use_bias=False, kernel_regularizer=regularizer))
            layers.append(tf.keras.layers.Conv2D(channels_per_subnet[i], kernel_size=3, activation="relu",
                                                 padding="same", use_bias=False, kernel_regularizer=regularizer))

    layers.append(tf.keras.layers.AveragePooling2D(pool_size=8))
    layers.append(tf.keras.layers.Flatten())
    layers.append(tf.keras.layers.Dense(units=10, activation="softmax", kernel_regularizer=regularizer,
                                        bias_regularizer=regularizer))

    return tf.keras.Sequential(layers)


if __name__ == '__main__':
    # train network
    train_graph = tf.Graph()
    train_sess = tf.Session(graph=train_graph)

    tf.keras.backend.set_session(train_sess)
    with train_graph.as_default():

        model = build_keras_model(args)

        data_gen = KDDataGenerator(args.train_val_set, args.batch_size, None, 1)

        # learning rate

        def step_decay(epoch_index, current_lr, epochs_with_decay=json.loads(args.epoch_step),
                       decay_ratio=args.lr_decay_ratio):
            if epoch_index + 1 in epochs_with_decay:
                return current_lr * decay_ratio
            else:
                return current_lr

        optimizer = SGD(lr=args.learning_rate, momentum=args.momentum)

        callbacks = [LearningRateScheduler(step_decay)]

        quant_delay = int(args.epochs * 4 / 5 * (45000 if args.train_val_set else 50000) / args.batch_size)
        # quant delay is set to 80% of training time
        tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=quant_delay)
        train_sess.run(tf.global_variables_initializer())

        # training
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['acc'])

        tf_dir = os.path.join(LOGS_DIR, args.save_file + "_val" if args.train_val_set else args.save_file)
        callbacks.append(LRTensorBoard(log_dir=tf_dir, batch_size=args.batch_size, write_graph=True))

        model.fit_generator(data_gen,
                            epochs=args.epochs,
                            validation_data=data_gen.get_val_data(),  # Take care that validation loss is meaningless when
                            # using KD (I'm forced to compute it by keras <3 <3 <3)
                            workers=args.workers,
                            callbacks=callbacks,
                            verbose=2)

        tf.contrib.quantize.create_eval_graph()

        if args.get_tf_lite:
            tflite_file = os.path.join(MODELS_DIR, f"{args.save_file}.tflite")
            tmp_folder = os.path.join(args.tmp_folder, args.save_file)
            tmp_pb_file = os.path.join(args.tmp_folder, f"{args.save_file}.pb")

            # save graph and checkpoints
            saver = tf.train.Saver()
            saver.save(train_sess, tmp_folder)

    if args.get_tf_lite:
        eval_graph = tf.Graph()
        eval_sess = tf.Session(graph=eval_graph)

        tf.keras.backend.set_session(eval_sess)
        with eval_sess:
            with eval_graph.as_default():
                eval_model = build_keras_model(args)
                tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
                eval_graph_def = eval_graph.as_graph_def()
                saver = tf.train.Saver()
                saver.restore(eval_sess, tmp_folder)

                frozen_graph_def = tf.graph_util.convert_variables_to_constants(
                    eval_sess,
                    eval_graph_def,
                    [eval_model.output.op.name]
                )

                with open(tmp_pb_file, 'wb') as f:
                    f.write(frozen_graph_def.SerializeToString())

        print("-----------")
        print("running tf_convert")
        print("-----------")
        command_line = f"tflite_convert --output_file={tflite_file} --graph_def_file={tmp_pb_file} \
        --inference_type=QUANTIZED_UINT8 --input_arrays=conv2d_input --output_arrays={eval_model.output.op.name} \
        --mean_values=128 --std_dev_values=64"
        print(subprocess.check_output(command_line, shell=True, executable='/bin/bash'))
