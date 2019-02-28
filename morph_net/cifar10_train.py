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

import tensorflow as tf
import pickle

from morph_net import cifar10
from morph_net import wideresnet
from morph_net.network_regularizers.flop_regularizer import GammaFlopsRegularizer

tf.app.flags.DEFINE_string('train_dir', '/home/nathan/Documents/TFE/morph_net/checkpoints',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_epochs', 200, """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10, """How often to log results to the console.""")
tf.app.flags.DEFINE_float('my_gamma_threshold', 1e-3, """The BN treshold under which a channel is considered pruned""")
tf.app.flags.DEFINE_float('regularizer_strength', 4e-9,
                          """the strenght of the regularizer in the loss term""")
tf.app.flags.DEFINE_string('pickle_file_name', 'pickle/res-40-2-rs=4e-9',
                           'the strenght of the regularizer in the loss term')

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
        logits = wideresnet.wideresnet_concat(40, 2, images, num_classes=10, drop_rate=0.0)

        network_reg = GammaFlopsRegularizer([logits.op], FLAGS.my_gamma_threshold)

        # Calculate loss.
        loss = cifar10.loss(logits, labels, network_reg)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = cifar10.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                save_checkpoint_secs=10000,
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            i = 0
            while i < FLAGS.max_epochs * cifar10.num_batches_per_epoch:
                mon_sess.run(train_op)
                i += 1

            # list the different operations
            for op in tf.get_default_graph().get_operations():
                name = str(op.name)
                layers_dict = {}
                if ("Conv" in name or "Skip" in name) and "Conv2D" in name and "gradients" not in name:
                    vec = mon_sess.run(network_reg._opreg_manager.get_regularizer(op).alive_vector)
                    channels_ratio = (sum(vec), len(vec))
                    print(name, ": %d/%d" % channels_ratio)
                    layers_dict[name] = channels_ratio

                with open(FLAGS.pickle_file_name, 'wb') as file:
                    pickle.dump(layers_dict, file)


def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
