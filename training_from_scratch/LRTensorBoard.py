from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard


class LRTensorBoard(TensorBoard):
    """callback to get learning rate on tensorboard, comes from
    https://stackoverflow.com/questions/49127214/keras-how-to-output-learning-rate-onto-tensorboard"""
    def __init__(self, log_dir, batch_size, write_graph):
        super().__init__(log_dir=log_dir, batch_size=batch_size, write_graph=write_graph)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)