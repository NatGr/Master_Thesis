"""quantizes a pretrained model, need tensorflow 2 nightly (as of early june 2019) to run
the interpreter ndo not work and we cannot manage to make the .tflite file run on tensorflow."""
from tensorflow.keras.datasets import cifar10
import numpy as np
import argparse
import os
from tensorflow import lite
import tensorflow as tf
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser(description='converts a keras network to a quantized tf_lite one')
parser.add_argument('--keras_file', default='saveto', type=str, help='name to use for this file')

args = parser.parse_args()
print(args)

KERAS_DIR = 'tmp_models'
TF_LITE_DIR = 'tf_lite_models'

if __name__ == '__main__':
    tmp_keras_file = os.path.join(KERAS_DIR, f"{args.keras_file}.h5")
    tflite_file = os.path.join(TF_LITE_DIR, f"{args.keras_file}_quantized.tflite")

    (x_calibrate, _), (x_test, y_test) = cifar10.load_data()
    channel_wise_mean = np.reshape(np.array([125.3, 123.0, 113.9]), (1, 1, 1, -1))
    channel_wise_std = np.reshape(np.array([63.0, 62.1, 66.7]), (1, 1, 1, -1))
    x_test = ((x_test - channel_wise_mean) / channel_wise_std).astype(np.float32)
    x_calibrate = tf.cast(((x_calibrate - channel_wise_mean) / channel_wise_std), tf.float32)


    # Convert to TensorFlow Lite model.
    model = load_model(tmp_keras_file)
    converter = lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [lite.Optimize.DEFAULT]

    calibration_ds = tf.data.Dataset.from_tensor_slices(x_calibrate).batch(1)  # turns into a dataset composed of
    # batches of size 1

    def representative_data_gen():
        for input_value in calibration_ds.take(50000):  # it does not cost anything to use the full dataset
            yield [input_value]
    converter.representative_dataset = representative_data_gen

    tflite_model = converter.convert()

    with open(tflite_file, "wb") as file:
        file.write(tflite_model)

    # computes accuracy
    interpreter = lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()

    input_layer = interpreter.get_input_details()[0]['index']
    output_layer = interpreter.get_output_details()[0]['index']

    correct_values = 0

    for i in range(x_test.shape[0]):
        # Test model on random input data.

        input_data = x_test[i:i+1, :, :, :]
        interpreter.set_tensor(input_layer, input_data)

        interpreter.invoke()

        output = interpreter.get_tensor(output_layer)

        if np.argmax(output) == y_test[i, 0]:
            correct_values += 1

    print(f"accuracy of {correct_values/100} %")

