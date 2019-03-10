# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""adapted from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_train.py
Routine for decoding the CIFAR-10 binary file format."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os

import tensorflow as tf
import pickle

from morph_net import cifar10
from morph_net import wideresnet
from morph_net.network_regularizers.flop_regularizer import GammaFlopsRegularizer, GroupLassoFlopsRegularizer

tf.app.flags.DEFINE_string('base_dir', '/home/nathan/Documents/TFE/morph_net',
                           """parent directory of train_dir""")
tf.app.flags.DEFINE_string('train_dir', 'checkpoints5e-8_tresh=1e-3_Group',
                           """Directory where to write event logs and checkpoint. (relative to base_dir)""")
tf.app.flags.DEFINE_integer('max_epochs', 200, """Number of epochs to run.""")
tf.app.flags.DEFINE_float('threshold', 1e-3, """The BN treshold under which a channel is considered pruned""")
tf.app.flags.DEFINE_float('regularizer_strength', 5e-8, """the strenght of the regularizer in the loss term""")
tf.app.flags.DEFINE_string('pickle_file_name', 'res-40-2-rs=5e-8-tresh=1e-3_Group.pickle',
                           'the strenght of the regularizer in the loss term')
tf.app.flags.DEFINE_boolean('use_gamma_reg', False,
                            """uses GammaFlopsRegularizer, if set to false, we will use GroupLassoFlopsRegularizer""")
tf.app.flags.DEFINE_string('channels_file_name', None,
                           """the name of the pickle file containing the number of channels 
we will use for each layer if we are retraining on an already pruned architacture""")
tf.app.flags.DEFINE_float('mult_fact', 1.8, """when using a preloaded number of channels, the mult factor to apply""")

FLAGS = tf.app.flags.FLAGS


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        depth = 40
        widening_factor = 2
        if FLAGS.channels_file_name is None:
            logits = wideresnet.wideresnet_concat(depth, widening_factor, images, num_classes=10, drop_rate=0.0)
        else:
            file = open(os.path.join('morph_net', 'pickle', FLAGS.channels_file_name), 'rb')
            dict_channels = pickle.load(file)
            channels_dict = {key: int(val*FLAGS.mult_fact) for key, (val, _) in dict_channels.items()}

            logits = wideresnet.wideresnet_concat(depth, widening_factor, images, num_classes=10, drop_rate=0.0,
                                                  channels_dict=channels_dict)

        if FLAGS.use_gamma_reg:
            network_reg = GammaFlopsRegularizer([logits.op], FLAGS.threshold)
        else:
            network_reg = GroupLassoFlopsRegularizer([logits.op], FLAGS.threshold)

        # Calculate loss.
        loss = cifar10.loss(logits, labels, network_reg)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = cifar10.train(loss, global_step)

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=os.path.join(FLAGS.base_dir, FLAGS.train_dir),
                hooks=[tf.train.NanTensorHook(loss)],
                save_checkpoint_secs=10000) as mon_sess:
            i = 0
            while i < FLAGS.max_epochs * cifar10.num_batches_per_epoch:
                mon_sess.run(train_op)
                i += 1

            # get back the number of pruned channels for every layer, the only way I found to select the operations
            # of interest were to list them all and then to perform the name filtering below, additionally to being
            # extremely ugly, this might break as soon as the name of the pruned layers do not contain "Conv" or "Skip"
            # or if a conv layer named "Conv"/"Skip" is not pruned :(
            layers_dict = {}
            for op in tf.get_default_graph().get_operations():
                name = str(op.name)
                if ("Conv" in name or "Skip" in name) and "Conv2D" in name and "gradients" not in name:
                    name = name[:-7]  # removes "/Conv2D" (hopefully they won't change that name in a future
                    # version of tf)
                    alive_channels = network_reg._opreg_manager.get_regularizer(op).alive_vector
                    vec = mon_sess.run(alive_channels)
                    channels_ratio = (sum(vec), len(vec))
                    print(name, ": %d/%d" % channels_ratio)
                    layers_dict[name] = channels_ratio

            with open(os.path.join('morph_net', 'pickle', FLAGS.pickle_file_name), 'wb') as file:
                pickle.dump(layers_dict, file)


def main(args=None):
    cifar10.maybe_download_and_extract()
    dir = os.path.join(FLAGS.base_dir, FLAGS.train_dir)
    if tf.gfile.Exists(dir):
        tf.gfile.DeleteRecursively(dir)
    tf.gfile.MakeDirs(dir)
    train()


if __name__ == '__main__':
    tf.app.run()
