from tensorflow import lite
from tensorflow.keras.datasets import cifar10
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Evaluation the test error of a tf_lite model on the test set')
parser.add_argument('--tf_lite_file', type=str, help='name of the file containing the model to use (no extension)')

args = parser.parse_args()
print(args)


if __name__ == '__main__':
    (_, _), (x_test, y_test) = cifar10.load_data()
    channel_wise_mean = np.reshape(np.array([125.3, 123.0, 113.9]), (1, 1, 1, -1))
    channel_wise_std = np.reshape(np.array([63.0, 62.1, 66.7]), (1, 1, 1, -1))
    x_test = ((x_test - channel_wise_mean) / channel_wise_std).astype(np.float32)

    interpreter = lite.Interpreter(model_path=f"{args.tf_lite_file}.tflite")
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

