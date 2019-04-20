"""trains a network for a given number of epochs and then saves it as a tf-lite model"""
import argparse
import json
import os
import pickle
import numpy as np
import cv2
from math import cos, pi
from tensorflow import lite
from albumentations import Compose, PadIfNeeded, RandomCrop, HorizontalFlip
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator, LearningRateScheduler
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD
from models.wideresnet import build_wrn


parser = argparse.ArgumentParser(description='Training and saving in tf_lite')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--save_file', default='saveto', type=str, help='name to use for this file')
parser.add_argument('--channels_pickle', type=str, help='name of the pickle file containing the number of channels for '
                                                        'each layers')
parser.add_argument('--tmp_folder', default='/dev/shm/tmp_models', type=str,
                    help='folder in which to create the tmp files (for tf-lite conversion), by default, uses linux '
                         'tmpfs file system')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true',
                    help='registers the validation loss and accuracy on tensorboard')
parser.set_defaults(tensorboard=False)


# Learning specific arguments
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--learning_rate', default=.1, type=float, help='initial learning rate')
parser.add_argument('--lr_type', default='multistep', type=str, help='learning rate strategy (default: multistep)',
                    choices=['cosine', 'multistep'])
parser.add_argument('--epochs', default=200, type=int, help='no. epochs')
parser.add_argument('--epoch_step', default='[60,120,160]', type=str, help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay factor')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=0.0005, type=float, help='weight decay')

# Net specific
parser.add_argument('--net', choices=['res'], default='res')
parser.add_argument('--depth', '-d', default=40, type=int, metavar='D', help='depth of network')

args = parser.parse_args()

LOGS_DIR = 'tf_logs'
MODELS_DIR = 'tf_lite_models'
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

with open(f"{args.channels_pickle}.pickle", 'rb') as file:
    channels_dict = pickle.load(file)

# network
inputs = Input(32, 32, 3)
if args.net == 'res':
    model = build_wrn(inputs, args.depth, channels_dict)
else:
    raise ValueError('pick a valid net')

if __name__ == '__main__':
    # data processing
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # normalizes using train_set data
    channel_wise_mean = np.reshape(np.array([125.3, 123.0, 113.9]), (1, 1, 1, -1))
    channel_wise_std = np.reshape(np.array([63.0, 62.1, 66.7]), (1, 1, 1, -1))
    x_train = (x_train - channel_wise_mean) / channel_wise_std
    x_test = (x_test - channel_wise_mean) / channel_wise_std

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # data augmentation
    augmentation = Compose([PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
                            RandomCrop(32, 32, p=1.0),
                            HorizontalFlip(p=0.5)])

    def real_data_augmentation_library(image, augmentation=augmentation):
        import pdb
        pdb.set_trace()
        augmented = augmentation(image=image)
        return augmented['image']
    datagen = ImageDataGenerator(preprocessing_function=real_data_augmentation_library)

    # training
    model.compile(optimizer=SGD(lr=args.learning_rate, momentum=args.momentum, decay=args.weight_decay),
                  loss='categorical_crossentropy',
                  metrics=['loss', 'acc'])

    if args.lr_type == 'cosine':
        def step_decay(epoch_index, _, total_num_epochs=args.epochs, init_lr=args.learning_rate):
            """takes an epoch index as input (integer, indexed from 0) and current learning
            rate and returns a new learning rate as output (float)."""
            decay = 0.5 * (1 + cos(pi * epoch_index / total_num_epochs))
            return init_lr * decay

    else:  # multistep
        def step_decay(epoch_index, current_lr, epochs_with_decay=json.loads(args.epoch_step),
                       decay_ratio=args.lr_decay_ratio):
            if epoch_index + 1 in epochs_with_decay:
                return current_lr * decay_ratio
            else:
                return current_lr

    callbacks = [LearningRateScheduler(step_decay)]
    if args.tensorboard:
        callbacks.append(TensorBoard(log_dir=LOGS_DIR, batch_size=args.batch_size, write_graph=True))

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=args.batch_size),
                        epochs=args.epochs,
                        validation_data=(x_test, y_test),
                        workers=args.workers,
                        callbacks=callbacks)

    # save it to tf_lite
    tmp_keras_file = os.path.join(args.tmp_folder, f"{args.save_file}.h5")
    tflite_file = os.path.join(MODELS_DIR, f"{args.save_file}.tflite")
    save_model(model, tmp_keras_file)

    # Convert to TensorFlow Lite model.
    converter = lite.TFLiteConverter.from_keras_model_file(tmp_keras_file)
    tflite_model = converter.convert()
    with open(tflite_file, "wb") as file:
        file.write(tflite_model)
