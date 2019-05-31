from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from albumentations import Compose, PadIfNeeded
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import numpy as np
import cv2


class KDDataGenerator(Sequence):
    """custom data genarator, this was built to be able to memoize the predictions of a model while using mild data
    augmentation. To do so, we have to use a trained model to get its prediction to all the thinkable inputs, i.e.
    128 (possible data augmentations) * 50000 (test set images)"""
    def __init__(self, train_val_set, batch_size, pred_teacher=None, temperature=None):
        """ constructor

        :param train_val_set: wether to train on 90% of the training set and use the rest as validation or use the
        train and test set
        :param batch_size: the abtch size to use
        :param pred_teacher: the array storing the predictions of the teacher, if None, no KD is used
        :param temperature: the temperature to use for training, must be specified if pred_teacher is not None
        """
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        if train_val_set:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train, self.y_train,
                                                                                    test_size=0.1,
                                                                                    stratify=self.y_train,
                                                                                    random_state=17)
        self.train_val_set = train_val_set
        self.batch_size = batch_size
        self.pred_teacher = pred_teacher
        self.temperature = temperature

        self.images_offsets = np.arange(self.x_train.shape[0])  # remember this so that teacher predictions are
        # correctly offsetted

        # normalizes using train_set data
        channel_wise_mean = np.reshape(np.array([125.3, 123.0, 113.9]), (1, 1, 1, -1))
        channel_wise_std = np.reshape(np.array([63.0, 62.1, 66.7]), (1, 1, 1, -1))
        self.x_train = (self.x_train - channel_wise_mean) / channel_wise_std
        self.x_test = (self.x_test - channel_wise_mean) / channel_wise_std

        # Convert class vectors to binary class matrices.
        self.y_train = to_categorical(self.y_train, num_classes=10)
        self.y_test = to_categorical(self.y_test, num_classes=10)

        # data augmentation
        self.pads = Compose([PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_REFLECT_101, p=1.0)])

    def on_epoch_end(self):
        """called at the end of each epoch"""
        permutation = np.random.permutation(self.x_train.shape[0])
        self.x_train = self.x_train[permutation, :, :, :]
        self.y_train = self.y_train[permutation, :]
        self.images_offsets = self.images_offsets[permutation]

    def __len__(self):
        """return the number of batches used to compute the number of steps per epoch"""
        return int(np.ceil(self.x_train.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        """return a batch, if we apply knowledge distilation, the third dimension of y_batch will contain the target as
        first element and the proba distribution of the teacher as second element"""
        x_batch = self.x_train[idx * self.batch_size:(idx + 1) * self.batch_size, :, :, :]
        y_batch = self.y_train[idx * self.batch_size:(idx + 1) * self.batch_size, :]

        # data augmentation
        augmentation = np.random.randint(128, size=x_batch.shape[0])
        x_batch = self._apply_data_augmentations(x_batch, augmentation)

        # Knowledge distilation
        if self.pred_teacher is not None:
            offsets = self.images_offsets[idx * self.batch_size:(idx + 1) * self.batch_size]
            logits_teacher = self.pred_teacher[offsets, augmentation, :]

            # turning logits into predictions (softmax outputs)
            pred_teacher = np.exp(logits_teacher / self.temperature)
            for i in range(offsets.shape[0]):
                pred_teacher[i, :] = pred_teacher[i, :] / np.sum(pred_teacher[i, :])
            y_batch = np.concatenate((y_batch, pred_teacher), axis=1)

        return x_batch, y_batch

    def _apply_data_augmentations(self, x_batch, augmentation):
        """applies the data augmentations corresponding to augmentation[i] to x_batch[i] and stores the result in a new
        array that is returned"""
        augmentation, x_offset = np.divmod(augmentation, 8)
        flip, y_offset = np.divmod(augmentation, 8)
        x_batch_new = np.empty(x_batch.shape)

        for i in range(x_batch.shape[0]):  # performs cropping and horizontal flip with the given parameters
            x_batch_pad = self.pads(image=x_batch[i, :, :, :])['image']
            if flip[i] == 0:
                x_batch_new[i, :, :, :] = x_batch_pad[x_offset[i]:x_offset[i] + 32, y_offset[i]:y_offset[i] + 32, :]
            else:
                x_batch_new[i, :, :, :] = cv2.flip(x_batch_pad[x_offset[i]:x_offset[i] + 32,
                                                   y_offset[i]:y_offset[i] + 32, :], 1)  # same as in the albumentations
                # flip function
        return x_batch_new

    def get_val_data(self):
        """returns a generator to the validation set
        Since we don't have KD predictions for the validation data, use the true labels instead, it is meaningless but
        keras forces to use KD loss when evaluating the data and I did not find a proper way to use conditionnals in the
        loss"""
        if self.pred_teacher is None:
            return self.x_test, self.y_test
        else:
            return self.x_test, np.concatenate((self.y_test, self.y_test), axis=1)

    def build_prediction_array(self, model, file_name):
        """builds the prediction array (containing the logits) for the model and saves it as a numpy array file
        (do not specify extension)"""
        reverse_offsets = np.argsort(self.images_offsets)  # get back the indices to recompute the initial ordering of
        # the array
        images = self.x_train[reverse_offsets, :, :, :]
        num_aug = 128
        model_pred = np.zeros((images.shape[0], num_aug, 10), dtype=np.float32)
        augmentation = np.arange(num_aug)

        for i in range(model_pred.shape[0]):
            x_batch = np.broadcast_to(images[i, :, :, :], (num_aug, images.shape[1], images.shape[2], images.shape[3]))
            x_batch = self._apply_data_augmentations(x_batch, augmentation)
            model_pred[i, :, :] = model.predict(x_batch, batch_size=num_aug)
            if (i + 1) % 1000 == 0:
                print(f"{i+1} images computed")

        np.savez_compressed(f"{file_name}{'_val' if self.train_val_set else ''}", pred=model_pred)
