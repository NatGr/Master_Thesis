"""trains a network for a given number of epochs and then saves it as a tf-lite model"""
import argparse
import json
import os
import pickle
from math import cos, pi
import numpy as np
from tensorflow import lite
from tensorflow.keras.layers import Input
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.regularizers import l2
from models.wideresnet import build_wrn
from models.effnet import build_effnet
from models.squeezenext import build_squeezenext
from models.mobilenetv1 import build_mobilenetv1
from models.mobilenetv2 import build_mobilenetv2
from models.shufflenetv1 import build_shufflenetv1
from models.shufflenetv2 import build_shufflenetv2
from models.nasnet import build_nasnet
from models.mnasnet import build_mnasnet
from LRTensorBoard import LRTensorBoard
from KDDataGenerator import KDDataGenerator
from losses import categorical_crossentropy_from_logits, knowledge_distillation_loss

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

# knowledge distillation arguments
parser.add_argument('--teacher_pred', default=None, type=str, help='name of the file containing the predictions made '
                                                                   'by the teacher (withtout extension and in folder '
                                                                   'teacher_pred)')
parser.add_argument('--temperature', default=5, type=int, help='temperature to use when doing KD')
parser.add_argument('--lambda_KD', default=.1, type=float, help='lambda factor attributed to the classical loss '
                                                                'when doing knowledge distillation')
parser.add_argument('--save_pred_table', action='store_true', help='saves the table containing all the possible '
                                                                   'predictions of the model')

# Learning specific arguments
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--learning_rate', default=.1, type=float, help='initial learning rate')
parser.add_argument('--lr_type', default='cosine', type=str, help='learning rate strategy (default: multistep)',
                    choices=['cosine', 'multistep', 'adam', 'rmsprop'])
parser.add_argument('--epochs', default=200, type=int, help='no. epochs')
parser.add_argument('--epoch_step', default='[60,120,160]', type=str, help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay factor')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=0.0001, type=float, help='weight decay')

# Net specific arguments
parser.add_argument('--net', choices=['resnet', 'effnet', 'squeezenext', 'mobilenetv1', 'mobilenetv2', 'shufflenetv1',
                                      'shufflenetv2', 'nasnet', 'mnasnet'])
parser.add_argument('--depth', '-d', default=40, type=int, help='depth of network')
parser.add_argument('--channels_pickle', default=None, type=str,
                    help='name of the pickle file containing the number of channels for each layers, only used with res')
parser.add_argument('--expansion_rate', default=2, type=int, help='effnet/mobilenetv2 expansion rate')
parser.add_argument('--num_groups', default=3, type=int, help='number of groups for shufflenetv1')
parser.add_argument('--width', default=32, type=int,
                    help='width of the first subnetwork, width are multiplied by 2 at each application of stride, '
                         'this argument is ignored for WRN if channels_pickle is specified')

parser.add_argument('--use_dropout', action='store_true',
                    help='whether to use a dropout of .2 before the final classification layer, '
                         'only affects the mobilenets, shufflenetv2 and mnasnet')
parser.add_argument('--use_5_5_filters', action='store_true',
                    help='whether to use 5*5 filters in the later subnetworks or only 3*3, '
                         'only affects mobilenetv1')
parser.add_argument('--se_factor', type=int, default=0,
                    help='reduction factor for the squeeze and excitation layer, no SE layer if set to 0, '
                         'only affects shufflenetv2, mnasnet, wideresnet and the mobilenets')
parser.add_argument('--add_skip', action='store_true',
                    help='whether to add skip connections to the network, only affects mobilenetv1 and shufflenetv2')

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

# network
inputs = Input((32, 32, 3))

if args.depth % 3 == 0:
    blocks_per_subnet = [int(args.depth / 3)] * 3
elif args.net != 'resnet':
    raise ValueError('Net must have a depth divisible by 3')

regularizer = l2(args.weight_decay)
channels_per_subnet = [args.width, args.width * 2, args.width * 4]

if args.net == 'effnet':
    channels_per_subnet.append(channels_per_subnet[-1] * 2)  # channels per subnet a bit different than for other
    # networks

if args.net == 'resnet':
    if args.channels_pickle is None:
        channels_dict = None
    else:
        with open(args.channels_pickle, 'rb') as file:
            channels_dict = pickle.load(file)
            if isinstance(next(iter(channels_dict.values())), tuple):  # in the case of morphnet,
                # values are (53, 128) instead of 53
                for key, value in channels_dict.items():
                    channels_dict[key] = value[0]

    model = build_wrn(inputs, args.depth, regularizer=regularizer, se_factor=args.se_factor,
                      channels_dict=channels_dict, width=args.width)

elif args.net == 'effnet':
    model = build_effnet(inputs, regularizer=regularizer, blocks_per_subnet=blocks_per_subnet,
                         expansion_rate=args.expansion_rate, channels_per_subnet=channels_per_subnet)

elif args.net == 'squeezenext':
    model = build_squeezenext(inputs, regularizer=regularizer, blocks_per_subnet=blocks_per_subnet,
                              channels_per_subnet=channels_per_subnet)

elif args.net == 'mobilenetv1':
    model = build_mobilenetv1(inputs, regularizer=regularizer, blocks_per_subnet=blocks_per_subnet,
                              channels_per_subnet=channels_per_subnet, use_5_5_filters=args.use_5_5_filters,
                              se_factor=args.se_factor, add_skip=args.add_skip, use_dropout=args.use_dropout)

elif args.net == 'mobilenetv2':
    model = build_mobilenetv2(inputs, regularizer=regularizer, blocks_per_subnet=blocks_per_subnet,
                              channels_per_subnet=channels_per_subnet, expansion_factor=args.expansion_rate,
                              use_dropout=args.use_dropout, se_factor=args.se_factor)
elif args.net == 'shufflenetv1':
    model = build_shufflenetv1(inputs, regularizer=regularizer, blocks_per_subnet=blocks_per_subnet,
                               channels_per_subnet=channels_per_subnet, num_groups=args.num_groups)
elif args.net == 'shufflenetv2':
    model = build_shufflenetv2(inputs, regularizer=regularizer, blocks_per_subnet=blocks_per_subnet,
                               channels_per_subnet=channels_per_subnet, se_factor=args.se_factor,
                               add_skip=args.add_skip, use_dropout=args.use_dropout)
elif args.net == 'nasnet':
    model = build_nasnet(inputs, regularizer, blocks_per_subnet=blocks_per_subnet,
                         channels_per_subnet=channels_per_subnet)
elif args.net == 'mnasnet':
    model = build_mnasnet(inputs, regularizer=regularizer, blocks_per_subnet=blocks_per_subnet,
                          channels_per_subnet=channels_per_subnet, expansion_factor=args.expansion_rate,
                          use_dropout=args.use_dropout, se_factor=args.se_factor)

else:
    raise ValueError('pick a valid net')

if __name__ == '__main__':
    if args.teacher_pred is None:
        pred_array = None
    else:
        try:
            with open(os.path.join(TEACHER_DIR, args.teacher_pred + '.npz'), 'rb') as file:
                pred_array = np.load(file)
                pred_array = pred_array["pred"]
        except FileNotFoundError:
            with open(os.path.join(TEACHER_DIR, args.teacher_pred + '.npy'), 'rb') as file:  # tries to open npy files
                # if npz fails (we did not used npz files initially)
                pred_array = np.load(file)

    data_gen = KDDataGenerator(args.train_val_set, args.batch_size, pred_array, args.temperature)

    # learning rate
    if args.lr_type == 'adam':
        optimizer = Adam(lr=args.learning_rate, beta_1=0.75)
        callbacks = []
    else:

        if args.lr_type == 'rmsprop':
            def step_decay(_, current_lr, decay_factor=args.lr_decay_ratio):
                """multiplies the current leraning rate by decay_factor at each epoch"""
                return current_lr * decay_factor

            optimizer = RMSprop(lr=args.learning_rate, decay=args.lr_decay_ratio)

        elif args.lr_type == 'cosine':
            def step_decay(epoch_index, _, total_num_epochs=args.epochs, init_lr=args.learning_rate):
                """takes an epoch index as input (integer, indexed from 0) and current learning
                rate and returns a new learning rate as output (float)."""
                decay = 0.5 * (1 + cos(pi * epoch_index / total_num_epochs))
                return init_lr * decay

            optimizer = SGD(lr=args.learning_rate, momentum=args.momentum, nesterov=True)

        else:  # multistep
            def step_decay(epoch_index, current_lr, epochs_with_decay=json.loads(args.epoch_step),
                           decay_ratio=args.lr_decay_ratio):
                if epoch_index + 1 in epochs_with_decay:
                    return current_lr * decay_ratio
                else:
                    return current_lr

            optimizer = SGD(lr=args.learning_rate, momentum=args.momentum)

        callbacks = [LearningRateScheduler(step_decay)]

    # training
    if args.teacher_pred is None:
        loss = categorical_crossentropy_from_logits
    else:
        def loss(target, output):
            return knowledge_distillation_loss(target, output, args.lambda_KD, args.temperature, num_classes=10)

    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    tf_dir = os.path.join(LOGS_DIR, args.save_file + "_val" if args.train_val_set else args.save_file)
    callbacks.append(LRTensorBoard(log_dir=tf_dir, batch_size=args.batch_size, write_graph=False))

    model.fit_generator(data_gen,
                        epochs=args.epochs,
                        validation_data=data_gen.get_val_data(),  # Take care that validation loss is meaningless when
                        # using KD (I'm forced to compute it by keras <3 <3 <3)
                        workers=args.workers,
                        callbacks=callbacks,
                        verbose=2)

    if args.save_pred_table:  # in that case we save the predictions on the entire dataset
        file_name = os.path.join(TEACHER_DIR, args.save_file)
        data_gen.build_prediction_array(model, file_name)

    if args.get_tf_lite:
        tflite_file = os.path.join(MODELS_DIR, f"{args.save_file}.tflite")
        tmp_keras_file = os.path.join(args.tmp_folder, f"{args.save_file}.h5")

        save_model(model, tmp_keras_file, include_optimizer=False)

        # Convert to TensorFlow Lite model.
        converter = lite.TFLiteConverter.from_keras_model_file(tmp_keras_file)
        tflite_model = converter.convert()

        with open(tflite_file, "wb") as file:
            file.write(tflite_model)
