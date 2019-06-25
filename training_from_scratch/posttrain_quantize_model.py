"""quantizes a pretrained model, need tensorflow 2 nightly (as of early june 2019) to run
the interpreter ndo not work and we cannot manage to make the .tflite file run on tensorflow."""
import numpy as np
import argparse
import os
import tensorflow as tf

parser = argparse.ArgumentParser(description='converts a keras network to a quantized tf_lite one')
parser.add_argument('--keras_file', default='saveto', type=str, help='name to use for this file')

args = parser.parse_args()
print(args)

KERAS_DIR = 'tmp_models'
TF_LITE_DIR = 'tf_lite_models'

if __name__ == '__main__':
    tmp_keras_file = os.path.join(KERAS_DIR, f"{args.keras_file}.h5")
    tflite_file = os.path.join(TF_LITE_DIR, f"{args.keras_file}_quantized.tflite")

    (x_calibrate, y_calibrate), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    channel_wise_mean = np.reshape(np.array([125.3, 123.0, 113.9]), (1, 1, 1, -1))
    channel_wise_std = np.reshape(np.array([63.0, 62.1, 66.7]), (1, 1, 1, -1))
    x_test = ((x_test - channel_wise_mean) / channel_wise_std).astype(np.float32)


    # Convert to TensorFlow Lite model.
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(tmp_keras_file)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    x_calibrate = ((x_calibrate - channel_wise_mean) / channel_wise_std).astype(np.float32)

    def representative_data_gen():
        for i in range(50000):
            yield [x_calibrate[i: i + 1]]

    converter.representative_dataset = representative_data_gen

    tflite_model = converter.convert()


    with open(tflite_file, "wb") as file:
        file.write(tflite_model)

    # computes accuracy
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]


    def quantize(real_value):
        std, mean = input_detail['quantization']
        return (real_value / std + mean).astype(np.uint8)

    correct_values = 0

    for i in range(x_test.shape[0]):
        # Test model on random input data.
        if i % 1000 == 0:
            print(f"{i} samples treated")

        input_data = quantize(x_test[i:i+1, :, :, :])
        interpreter.set_tensor(input_detail['index'], input_data)

        interpreter.invoke()

        output = interpreter.get_tensor(output_detail['index'])

        if np.argmax(output) == y_test[i, 0]:
            correct_values += 1

    print(f"accuracy of {correct_values/100} %")

