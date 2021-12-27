#  Copyright (c) 2021 Arm Limited. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
This script will provide you with an example of how to perform post-training quantization in TensorFlow.

The output from this example will be a TensorFlow Lite model file where weights and activations are quantized to 8bit
integer values.

Quantization helps reduce the size of your models and is necessary for running models on certain hardware such as Arm
Ethos NPU.

In addition to quantizing weights, post-training quantization uses a calibration dataset to
capture the minimum and maximum values of all variable tensors in your model.
By capturing these ranges it is possible to fully quantize not just the weights of the model but also the activations.

Depending on the model you are quantizing there may be some accuracy loss, but for a lot of models the loss should
be minimal.

If you are targetting an Arm Ethos-U55 NPU then the output TensorFlow Lite file will also need to be passed through the Vela
compiler for further optimizations before it can be used.

For more information on using Vela see: https://git.mlplatform.org/ml/ethos-u/ethos-u-vela.git/about/
For more information on post-training quantization
see: https://www.tensorflow.org/lite/performance/post_training_integer_quant
"""

import pathlib
import os
import argparse
import numpy as np
import tensorflow as tf


from dataset.dataset import BevDataSet
from yolo_training_utils import create_model, get_data, YoloLoss


def post_training_quantize(keras_model, sample_data):
    """Quantize Keras model using post-training quantization with some sample data.

    TensorFlow Lite will have fp32 inputs/outputs and the model will handle quantizing/dequantizing.

    Args:
        keras_model: Keras model to quantize.
        sample_data: A numpy array of data to use as a representative dataset.

    Returns:
        Quantized TensorFlow Lite model.
    """

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    # We set the following converter options to ensure our model is fully quantized.
    # An error should get thrown if there is any ops that can't be quantized.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # To use post training quantization we must provide some sample data that will be used to
    # calculate activation ranges for quantization. This data should be representative of the data
    # we expect to feed the model and must be provided by a generator function.
    def generate_repr_dataset():
        for i in range(100):  # 100 samples is all we should need in this example.
            yield [np.expand_dims(sample_data[i], axis=0)]

    converter.representative_dataset = generate_repr_dataset
    tflite_model = converter.convert()

    return tflite_model


def evaluate_tflite_model(tflite_save_path, x_test, y_test, batch_size=8):
    """Calculate the accuracy of a TensorFlow Lite model using TensorFlow Lite interpreter.

    Args:
        tflite_save_path: Path to TensorFlow Lite model to test.
        x_test: numpy array of testing data.
        y_test: numpy array of testing labels (sparse categorical).
    """

    interpreter = tf.lite.Interpreter(model_path=str(tflite_save_path))

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    accuracy_count = 0
    num_test_images = len(y_test)
    loss = 0

    for i in range(num_test_images):
        output_batch = []
        label_batch = []
        for j in range(batch_size):
            label_batch.append(y_test[i])
            interpreter.set_tensor(input_details[0]['index'], x_test[i][np.newaxis, ...])
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_data = np.squeeze(output_data)
            output_batch.append(output_data)
        output_batch = tf.stack(output_batch)
        label_batch = np.asarray(label_batch)
        #output_data = np.squeeze(output_data)
        print("loss({}): {}".format(i, loss))
        print("type(label_batch): {}, type(output_batch): {}".format(type(label_batch), type(output_batch)))
        print("shape(label_batch): {}, shape(output_batch): {}".format(label_batch.shape, tf.shape(output_batch)))
        loss += yolo_loss(label_batch, output_batch)

    print(f"Test loss quantized: {loss / num_test_images:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_weights", type=str, default='False', help="Whether to load weights, True or False")
    parser.add_argument("--batch_size", type=int, default=8, help="Set the batch_size")
    parser.add_argument("--weights_path", type=str, default="./weights/...", help="Set the weights_path")
    parser.add_argument("--save_dir", type=str, default="./weights/", help="Dir to save weights")
    parser.add_argument("--gpu_id", type=str, default='0', help="Specify GPU device")
    parser.add_argument("--num_iter", type=int, default=16000, help="num_max_iter")
    parser.add_argument("--save_interval", type=int, default=2, help="Save once every two epochs")
    parser.add_argument("--data_path", type=str, default="kitti/image_dataset/", help="training data path")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    SCALE = 32
    GRID_W, GRID_H = 32, 24
    N_CLASSES = 8
    N_ANCHORS = 5
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH = GRID_H * SCALE, GRID_W * SCALE, 3
    batch_size = args.batch_size
    data_path = args.data_path

    #x_train, y_train, x_test, y_test = get_data(data_path=data_path)
    train_dataset = BevDataSet(data_set='train',
                                 mode='train',
                                 load_to_memory=False,
                                 datapath=data_path,
                                 batch_size=batch_size)
    test_dataset = BevDataSet(data_set='test',
                                mode='test',
                                flip=False,
                                aug_hsv=False,
                                random_scale=False,
                                load_to_memory=False,
                                datapath=data_path,
                                batch_size=batch_size)
    model = create_model()
    model.summary()
    # Compile and train the model in fp32 as normal.
    yolo_loss = YoloLoss(batch_size=batch_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=yolo_loss)

    model.fit(x=train_dataset, validation_data=test_dataset, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
    #model.fit_generator(generator=train_dataset, validation_data=test_dataset, epochs=5, verbose=1, shuffle=True)

    # Test the fp32 model accuracy.
    #test_loss, test_acc = model.evaluate(x=test_dataset)
    #print(f"Test loss: {test_loss:.3f}, Test accuracy float: {test_acc:.3f}")

    # Quantize and export the resulting TensorFlow Lite model to file.
    x_train, y_train = train_dataset.get_sample(100)
    print("x_train type: {}".format(type(x_train)))
    print("x_train shape: {}".format(tf.shape(x_train)))
    x_train = x_train.astype(np.float32)
    tflite_model = post_training_quantize(model, x_train)

    print("Created quantized model")
    tflite_models_dir = pathlib.Path('./yolo_conditioned_models/')
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    quant_model_save_path = tflite_models_dir / 'yolo_post_training_quant_model.tflite'
    with open(quant_model_save_path, 'wb') as f:
        f.write(tflite_model)

    print("Saved quantized model")
    # Test the quantized model accuracy. Save time by only testing a subset of the whole data.
    x_test, y_test = test_dataset.get_sample(batch_size)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    print("Start evaluating quantized model")
    evaluate_tflite_model(quant_model_save_path, x_test, y_test, batch_size=batch_size)
    print("Finished evaluating quantized model")

if __name__ == "__main__":
    main()
