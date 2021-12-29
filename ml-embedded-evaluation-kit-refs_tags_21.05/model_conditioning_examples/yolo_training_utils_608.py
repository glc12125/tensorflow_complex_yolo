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
Utility functions related to data and models that are common to all the model conditioning examples.
"""
import tensorflow as tf
import numpy as np

from dataset.dataset_608 import BevDataSet

SCALE = 32
GRID_W, GRID_H = 19, 19
N_CLASSES = 8
N_ANCHORS = 5
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH = GRID_H * SCALE, GRID_W * SCALE, 3
class_dict = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7
}

def get_data(data_path="kitti/image_dataset/"):
    """Downloads and returns the pre-processed data and labels for training and testing.

    Returns:
        Tuple of: (train data, train labels, test data, test labels)
    """

    ## To save time we use the MNIST dataset for this example.
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    ## Convolution operations require data to have 4 dimensions.
    ## We divide by 255 to help training and cast to float32 for TensorFlow.
    #x_train = (x_train[..., np.newaxis] / 255.0).astype(np.float32)
    #x_test = (x_test[..., np.newaxis] / 255.0).astype(np.float32)

    train_dataset = BevDataSet(data_set='train',
                                 mode='train',
                                 load_to_memory=False,
                                 datapath=data_path)
    test_dataset = BevDataSet(data_set='test',
                                mode='test',
                                flip=False,
                                aug_hsv=False,
                                random_scale=False,
                                load_to_memory=False,
                                datapath=data_path)

    x_train, y_train = train_dataset.get_all()
    x_test, y_test = test_dataset.get_all()

    return x_train, y_train, x_test, y_test


def max_pool_layer(size, stride, name):
    #x = tf.compat.v1.layers.max_pooling2d(x, size, stride, padding='SAME')
    #x = tf.keras.layers.MaxPool2D(size, stride, padding='same')(x)
    #return x
    return [tf.keras.layers.MaxPool2D(size, stride, padding='same')]


def conv_layer(kernel, depth, train_logical, name):
    #return [tf.keras.layers.Conv2D(depth, kernel, padding='SAME', kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), activation=tf.nn.relu6),tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=0.001, center=True, scale=True)]
    #return [tf.keras.layers.Conv2D(depth, kernel, padding='SAME', kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), activation=tf.nn.leaky_relu),tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=0.001, center=True, scale=True)]
    return [tf.keras.layers.Conv2D(depth, kernel, padding='SAME', kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")),tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=0.001, center=True, scale=True), tf.keras.layers.LeakyReLU(alpha=0.2)]

def slice_tensor(x, start, end=None):
    """
    Get tensor slices
    param x (array):
    param start (int):
    param end (int):
    return (array):
    """
    if end < 0:
        y = x[..., start:]
    else:
        if end is None:
            end = start
        y = x[..., start:end + 1]
    return y


def max_pool_layer_functional(x, size, stride, name):
    x = tf.keras.layers.MaxPool2D(size, stride, padding='same')(x)
    return x


def conv_layer_functional(x, kernel, depth, train_logical, name):

    x = tf.keras.layers.Conv2D(
        depth,
        kernel,
        padding='SAME',
        activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.BatchNormalization(
                                      momentum=0.9,
                                      epsilon=0.001,
                                      center=True,
                                      scale=True)(x)
    return x


class SpaceToDepth(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpaceToDepth, self).__init__(**kwargs)

    def call(self, x, block_size):
        return tf.nn.space_to_depth(x, block_size)

def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    return tf.nn.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.

    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])

def passthrough_layer_functional(a, b, kernel, depth, size, train_logical, name):
    b = conv_layer_functional(b, kernel, depth, train_logical, name)
    #space_to_depth = SpaceToDepth()
    #b = space_to_depth(b, block_size=size)
    #b = tf.keras.layers.Lambda(
    #    space_to_depth_x2,
    #    output_shape=space_to_depth_x2_output_shape,
    #    name='space_to_depth')(b)
    #y = tf.concat([a, b], axis=3)
    b = tf.nn.space_to_depth(b, block_size=2)
    y = tf.keras.layers.Concatenate(axis=3)([a, b])
    return y

def create_model_with_pass_through(train_logical=True):
    """Create and returns a simple Keras model for training MNIST.

    We will use a simple convolutional neural network for this example,
    but the model optimization methods employed should be compatible with a
    wide variety of CNN architectures such as Mobilenet and Inception etc.

    Returns:
        Uncompiled Keras model.
    """
    inputs = tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    x = conv_layer_functional(inputs, (3, 3), 24, train_logical, 'conv1')
    x = max_pool_layer_functional(x, (2, 2), (2, 2), 'maxpool1')
    x = conv_layer_functional(x, (3, 3), 48, train_logical, 'conv2')
    x = max_pool_layer_functional(x, (2, 2), (2, 2), 'maxpool2')

    x = conv_layer_functional(x, (3, 3), 64, train_logical, 'conv3')
    x = conv_layer_functional(x, (1, 1), 32, train_logical, 'conv4')
    x = conv_layer_functional(x, (3, 3), 64, train_logical, 'conv5')
    x = max_pool_layer_functional(x, (2, 2), (2, 2), 'maxpool5')

    x = conv_layer_functional(x, (3, 3), 128, train_logical, 'conv6')
    x = conv_layer_functional(x, (1, 1), 64, train_logical, 'conv7')
    x = conv_layer_functional(x, (3, 3), 128, train_logical, 'conv8')
    x = max_pool_layer_functional(x, (2, 2), (2, 2), 'maxpool8')

    # x = conv_layer_functional(x, (3, 3), 512, train_logical, 'conv9')
    # x = conv_layer_functional(x, (1, 1), 256, train_logical, 'conv10')
    x = conv_layer_functional(x, (3, 3), 512, train_logical, 'conv11')
    x = conv_layer_functional(x, (1, 1), 256, train_logical, 'conv12')
    passthrough = conv_layer_functional(x, (3, 3), 512, train_logical, 'conv13')
    x = max_pool_layer_functional(passthrough, (2, 2), (2, 2), 'maxpool13')

    # x = conv_layer_functional(x, (3, 3), 1024, train_logical, 'conv14')
    # x = conv_layer_functional(x, (1, 1), 512, train_logical, 'conv15')
    x = conv_layer_functional(x, (3, 3), 1024, train_logical, 'conv16')
    x = conv_layer_functional(x, (1, 1), 512, train_logical, 'conv17')
    x = conv_layer_functional(x, (3, 3), 1024, train_logical, 'conv18')

    x = passthrough_layer_functional(x, passthrough, (3, 3), 64, 2, train_logical,
                          'conv21')
    x = conv_layer_functional(x, (3, 3), 1024, train_logical, 'conv19')
    outputs = conv_layer_functional(x, (1, 1), N_ANCHORS * (7 + N_CLASSES), train_logical,
                   'conv20')  # x,y,w,l,re,im,conf + 8 class

    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="robok_3d_detection_model")
    return keras_model

def create_model(train_logical=True):
    """Create and returns a simple Keras model for training MNIST.

    We will use a simple convolutional neural network for this example,
    but the model optimization methods employed should be compatible with a
    wide variety of CNN architectures such as Mobilenet and Inception etc.

    Returns:
        Uncompiled Keras model.
    """
    model_layers = (
        [tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))] +
        conv_layer((3, 3), 24, train_logical, 'conv1') +
        max_pool_layer((2, 2), (2, 2), 'maxpool1') +
        conv_layer( (3, 3), 48, train_logical, 'conv2') +
        max_pool_layer((2, 2), (2, 2), 'maxpool2') +
        conv_layer((3, 3), 64, train_logical, 'conv3') +
        conv_layer((1, 1), 32, train_logical, 'conv4') +
        conv_layer((3, 3), 64, train_logical, 'conv5') +
        max_pool_layer((2, 2), (2, 2), 'maxpool5') +
        conv_layer((3, 3), 128, train_logical, 'conv6') +
        conv_layer((1, 1), 64, train_logical, 'conv7') +
        conv_layer((3, 3), 128, train_logical, 'conv8') +
        max_pool_layer((2, 2), (2, 2), 'maxpool8') +
        conv_layer((3, 3), 512, train_logical, 'conv11') +
        conv_layer((1, 1), 256, train_logical, 'conv12') +
        conv_layer( (3, 3), 512, train_logical, 'conv13') +
        max_pool_layer((2, 2), (2, 2), 'maxpool13') +
        conv_layer((3, 3), 1024, train_logical, 'conv16') +
        conv_layer((1, 1), 512, train_logical, 'conv17') +
        conv_layer((3, 3), 1024, train_logical, 'conv18') +
        conv_layer((3, 3), 1024, train_logical, 'conv19') +
        conv_layer((1, 1), N_ANCHORS * (7 + N_CLASSES), train_logical, 'conv20'))
    #    [tf.keras.layers.Reshape((GRID_H, GRID_W, N_ANCHORS, 7 + N_CLASSES))])
    #keras_model = tf.keras.models.Sequential([
    #    tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=(28, 28, 1), activation=tf.nn.relu),
    #    tf.keras.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu),
    #    tf.keras.layers.MaxPool2D(),
    #    tf.keras.layers.Conv2D(32, 3, padding='same', activation=tf.nn.relu),
    #    tf.keras.layers.MaxPool2D(),
    #    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
    #])
    keras_model = tf.keras.models.Sequential(model_layers)

    return keras_model

def yolo_loss_original(label, pred):
    #print("type(label): {}, type(pred): {}".format(type(label), type(pred)))
    #print("shape(label): {}, shape(pred): {}".format(label.shape, pred.shape))
    batch_size = 8
    #pred = tf.reshape(pred, shape=(-1, GRID_H, GRID_W, N_ANCHORS, 7 + N_CLASSES))
    #print("shape(pred) after reshape: {}".format(pred.shape))
    #print("batch_Size: {}".format(batch_size))
    mask = slice_tensor(label, 7, 7)
    #print("shape(mask): {}".format(mask.shape))
    label = slice_tensor(label, 0, 6)
    mask = tf.cast(tf.reshape(mask, shape=(-1, GRID_H, GRID_W, N_ANCHORS)),
                   tf.bool)
    #print("shape(mask) after reshape and cast: {}".format(mask.shape))
    with tf.name_scope('mask'):
        masked_label = tf.boolean_mask(label, mask)
        masked_pred = tf.boolean_mask(pred, mask)
        neg_masked_pred = tf.boolean_mask(pred, tf.logical_not(mask))
    with tf.name_scope('pred'):
        masked_pred_xy = tf.sigmoid(slice_tensor(masked_pred, 0, 1))
        masked_pred_wh = slice_tensor(masked_pred, 2, 3)
        masked_pred_re = 2 * tf.sigmoid(slice_tensor(masked_pred, 4, 4)) - 1
        masked_pred_im = 2 * tf.sigmoid(slice_tensor(masked_pred, 5, 5)) - 1
        masked_pred_o = tf.sigmoid(slice_tensor(masked_pred, 6, 6))

        masked_pred_no_o = tf.sigmoid(slice_tensor(neg_masked_pred, 6, 6))
        # masked_pred_c = tf.nn.sigmoid(slice_tensor(masked_pred, 7, -1))
        masked_pred_c = tf.nn.softmax(slice_tensor(masked_pred, 7, -1))

    # masked_pred_no_c = tf.nn.sigmoid(slice_tensor(neg_masked_pred, 7, -1))
    # print (masked_pred_c, masked_pred_o, masked_pred_no_o)

    with tf.name_scope('lab'):
        masked_label_xy = slice_tensor(masked_label, 0, 1)
        masked_label_wh = slice_tensor(masked_label, 2, 3)
        masked_label_re = slice_tensor(masked_label, 4, 4)
        masked_label_im = slice_tensor(masked_label, 5, 5)
        masked_label_class = slice_tensor(masked_label, 6, 6)
        masked_label_class_vec = tf.reshape(tf.one_hot(tf.cast(
            masked_label_class, tf.int32),
            depth=N_CLASSES),
            shape=(-1, N_CLASSES))
    with tf.name_scope('merge'):
        with tf.name_scope('loss_xy'):
            loss_xy = tf.reduce_sum(
                tf.square(masked_pred_xy - masked_label_xy)) / batch_size
        with tf.name_scope('loss_wh'):
            loss_wh = tf.reduce_sum(
                tf.square(masked_pred_wh - masked_label_wh)) / batch_size
        with tf.name_scope('loss_re'):
            loss_re = tf.reduce_sum(
                tf.square(masked_pred_re - masked_label_re)) / batch_size
        with tf.name_scope('loss_im'):
            loss_im = tf.reduce_sum(
                tf.square(masked_pred_im - masked_label_im)) / batch_size
        with tf.name_scope('loss_obj'):
            loss_obj = tf.reduce_sum(tf.square(masked_pred_o - 1)) / batch_size
        # loss_obj =  tf.reduce_sum(-tf.log(masked_pred_o+0.000001))*10
        with tf.name_scope('loss_no_obj'):
            loss_no_obj = tf.reduce_sum(
                tf.square(masked_pred_no_o)) * 0.5 / batch_size
        # loss_no_obj =  tf.reduce_sum(-tf.log(1-masked_pred_no_o+0.000001))
        with tf.name_scope('loss_class'):
            # loss_c = tf.reduce_sum(tf.square(masked_pred_c - masked_label_c_vec))
            loss_c = (tf.reduce_sum(-tf.math.log(masked_pred_c + 0.000001) * masked_label_class_vec)
                      + tf.reduce_sum(-tf.math.log(1 - masked_pred_c + 0.000001) * (1 - masked_label_class_vec))) / batch_size
            # + tf.reduce_sum(-tf.log(1 - masked_pred_no_c+0.000001)) * 0.1
    # loss = (loss_xy + loss_wh+ loss_re + loss_im+ lambda_coord*loss_obj) + lambda_no_obj*loss_no_obj + loss_c
    loss = (loss_xy + loss_wh + loss_re +
            loss_im) * 5 + loss_obj + loss_no_obj + loss_c
    #return loss, loss_xy, loss_wh, loss_re, loss_im, loss_obj, loss_no_obj, loss_c
    return loss

def yolo_loss(label, pred):
    print("type(label): {}, type(pred): {}".format(type(label), type(pred)))
    print("shape(label): {}, shape(pred): {}".format(label.shape, pred.shape))
    batch_size = 8
    pred = tf.reshape(pred, shape=(-1, GRID_H, GRID_W, N_ANCHORS, 7 + N_CLASSES))
    print("shape(pred) after reshape: {}".format(pred.shape))
    #print("batch_Size: {}".format(batch_size))
    mask = slice_tensor(label, 7, 7)
    print("shape(mask): {}".format(mask.shape))
    label = slice_tensor(label, 0, 6)
    mask = tf.cast(tf.reshape(mask, shape=(-1, GRID_H, GRID_W, N_ANCHORS)),
                   tf.bool)
    print("shape(mask) after reshape and cast: {}".format(mask.shape))
    with tf.name_scope('mask'):
        masked_label = tf.boolean_mask(label, mask)
        masked_pred = tf.boolean_mask(pred, mask)
        neg_masked_pred = tf.boolean_mask(pred, tf.logical_not(mask))
    with tf.name_scope('pred'):
        masked_pred_xy = tf.sigmoid(slice_tensor(masked_pred, 0, 1))
        masked_pred_wh = slice_tensor(masked_pred, 2, 3)
        masked_pred_re = 2 * tf.sigmoid(slice_tensor(masked_pred, 4, 4)) - 1
        masked_pred_im = 2 * tf.sigmoid(slice_tensor(masked_pred, 5, 5)) - 1
        masked_pred_o = tf.sigmoid(slice_tensor(masked_pred, 6, 6))

        masked_pred_no_o = tf.sigmoid(slice_tensor(neg_masked_pred, 6, 6))
        masked_pred_c = tf.nn.sigmoid(slice_tensor(masked_pred, 7, -1))

    # masked_pred_no_c = tf.nn.sigmoid(slice_tensor(neg_masked_pred, 7, -1))
    # print (masked_pred_c, masked_pred_o, masked_pred_no_o)

    with tf.name_scope('lab'):
        masked_label_xy = slice_tensor(masked_label, 0, 1)
        masked_label_wh = slice_tensor(masked_label, 2, 3)
        masked_label_re = slice_tensor(masked_label, 4, 4)
        masked_label_im = slice_tensor(masked_label, 5, 5)
        masked_label_class = slice_tensor(masked_label, 6, 6)
        masked_label_class_vec = tf.reshape(tf.one_hot(tf.cast(
            masked_label_class, tf.int32),
            depth=N_CLASSES),
            shape=(-1, N_CLASSES))
    with tf.name_scope('merge'):
        with tf.name_scope('loss_xy'):
            loss_xy = tf.reduce_sum(
                tf.square(masked_pred_xy - masked_label_xy)) / batch_size
        with tf.name_scope('loss_wh'):
            loss_wh = tf.reduce_sum(
                tf.square(masked_pred_wh - masked_label_wh)) / batch_size
        with tf.name_scope('loss_re'):
            loss_re = tf.reduce_sum(
                tf.square(masked_pred_re - masked_label_re)) / batch_size
        with tf.name_scope('loss_im'):
            loss_im = tf.reduce_sum(
                tf.square(masked_pred_im - masked_label_im)) / batch_size
        with tf.name_scope('loss_obj'):
            loss_obj = tf.reduce_sum(tf.square(masked_pred_o - 1)) / batch_size
        # loss_obj =  tf.reduce_sum(-tf.log(masked_pred_o+0.000001))*10
        with tf.name_scope('loss_no_obj'):
            loss_no_obj = tf.reduce_sum(
                tf.square(masked_pred_no_o)) * 0.5 / batch_size
        # loss_no_obj =  tf.reduce_sum(-tf.log(1-masked_pred_no_o+0.000001))
        with tf.name_scope('loss_class'):
            # loss_c = tf.reduce_sum(tf.square(masked_pred_c - masked_label_c_vec))
            loss_c = (tf.reduce_sum(-tf.math.log(masked_pred_c + 0.000001) * masked_label_class_vec)
                      + tf.reduce_sum(-tf.math.log(1 - masked_pred_c + 0.000001) * (1 - masked_label_class_vec))) / batch_size
            # + tf.reduce_sum(-tf.log(1 - masked_pred_no_c+0.000001)) * 0.1
    # loss = (loss_xy + loss_wh+ loss_re + loss_im+ lambda_coord*loss_obj) + lambda_no_obj*loss_no_obj + loss_c
    loss = (loss_xy + loss_wh + loss_re +
            loss_im) * 5 + loss_obj + loss_no_obj + loss_c
    #return loss, loss_xy, loss_wh, loss_re, loss_im, loss_obj, loss_no_obj, loss_c
    print("loss_xy: {}".format(loss_xy))
    print("loss_wh: {}".format(loss_wh))
    print("loss_re: {}".format(loss_re))
    print("loss_im: {}".format(loss_im))
    print("loss_obj: {}".format(loss_obj))
    print("loss_no_obj: {}".format(loss_no_obj))
    print("loss_c: {}".format(loss_c))
    print("loss: {}".format(loss))
    return loss

class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, batch_size = 8):
        super().__init__()
        self.batch_size = batch_size
    def call(self, label, pred):
        #print("type(label): {}, type(pred): {}".format(type(label), type(pred)))
        #print("shape(label): {}, shape(pred): {}".format(label.shape, pred.shape))
        pred = tf.reshape(pred, shape=(-1, GRID_H, GRID_W, N_ANCHORS, 7 + N_CLASSES))
        #print("shape(pred) after reshape: {}".format(pred.shape))
        mask = slice_tensor(label, 7, 7)
        #print("shape(mask): {}".format(mask.shape))
        label = slice_tensor(label, 0, 6)
        mask = tf.cast(tf.reshape(mask, shape=(-1, GRID_H, GRID_W, N_ANCHORS)),
                       tf.bool)
        #print("shape(mask) after reshape and cast: {}".format(mask.shape))
        with tf.name_scope('mask'):
            masked_label = tf.boolean_mask(label, mask)
            masked_pred = tf.boolean_mask(pred, mask)
            neg_masked_pred = tf.boolean_mask(pred, tf.logical_not(mask))
        with tf.name_scope('pred'):
            masked_pred_xy = tf.sigmoid(slice_tensor(masked_pred, 0, 1))
            masked_pred_wh = slice_tensor(masked_pred, 2, 3)
            masked_pred_re = 2 * tf.sigmoid(slice_tensor(masked_pred, 4, 4)) - 1
            masked_pred_im = 2 * tf.sigmoid(slice_tensor(masked_pred, 5, 5)) - 1
            masked_pred_o = tf.sigmoid(slice_tensor(masked_pred, 6, 6))

            masked_pred_no_o = tf.sigmoid(slice_tensor(neg_masked_pred, 6, 6))
            masked_pred_c = tf.nn.sigmoid(slice_tensor(masked_pred, 7, -1))

        # masked_pred_no_c = tf.nn.sigmoid(slice_tensor(neg_masked_pred, 7, -1))
        # print (masked_pred_c, masked_pred_o, masked_pred_no_o)

        with tf.name_scope('lab'):
            masked_label_xy = slice_tensor(masked_label, 0, 1)
            masked_label_wh = slice_tensor(masked_label, 2, 3)
            masked_label_re = slice_tensor(masked_label, 4, 4)
            masked_label_im = slice_tensor(masked_label, 5, 5)
            masked_label_class = slice_tensor(masked_label, 6, 6)
            masked_label_class_vec = tf.reshape(tf.one_hot(tf.cast(
                masked_label_class, tf.int32),
                depth=N_CLASSES),
                shape=(-1, N_CLASSES))
        with tf.name_scope('merge'):
            with tf.name_scope('loss_xy'):
                loss_xy = tf.reduce_sum(
                    tf.square(masked_pred_xy - masked_label_xy)) / self.batch_size
            with tf.name_scope('loss_wh'):
                loss_wh = tf.reduce_sum(
                    tf.square(masked_pred_wh - masked_label_wh)) / self.batch_size
            with tf.name_scope('loss_re'):
                loss_re = tf.reduce_sum(
                    tf.square(masked_pred_re - masked_label_re)) / self.batch_size
            with tf.name_scope('loss_im'):
                loss_im = tf.reduce_sum(
                    tf.square(masked_pred_im - masked_label_im)) / self.batch_size
            with tf.name_scope('loss_obj'):
                loss_obj = tf.reduce_sum(tf.square(masked_pred_o - 1)) / self.batch_size
            # loss_obj =  tf.reduce_sum(-tf.log(masked_pred_o+0.000001))*10
            with tf.name_scope('loss_no_obj'):
                loss_no_obj = tf.reduce_sum(
                    tf.square(masked_pred_no_o)) * 0.5 / self.batch_size
            # loss_no_obj =  tf.reduce_sum(-tf.log(1-masked_pred_no_o+0.000001))
            with tf.name_scope('loss_class'):
                # loss_c = tf.reduce_sum(tf.square(masked_pred_c - masked_label_c_vec))
                loss_c = (tf.reduce_sum(-tf.math.log(masked_pred_c + 0.000001) * masked_label_class_vec)
                          + tf.reduce_sum(-tf.math.log(1 - masked_pred_c + 0.000001) * (1 - masked_label_class_vec))) / self.batch_size
                # + tf.reduce_sum(-tf.log(1 - masked_pred_no_c+0.000001)) * 0.1
        # loss = (loss_xy + loss_wh+ loss_re + loss_im+ lambda_coord*loss_obj) + lambda_no_obj*loss_no_obj + loss_c
        loss = (loss_xy + loss_wh + loss_re +
                loss_im) * 5 + loss_obj + loss_no_obj + loss_c
        #return loss, loss_xy, loss_wh, loss_re, loss_im, loss_obj, loss_no_obj, loss_c
        #print("loss_xy: {}".format(loss_xy))
        #print("loss_wh: {}".format(loss_wh))
        #print("loss_re: {}".format(loss_re))
        #print("loss_im: {}".format(loss_im))
        #print("loss_obj: {}".format(loss_obj))
        #print("loss_no_obj: {}".format(loss_no_obj))
        #print("loss_c: {}".format(loss_c))
        #print("loss: {}".format(loss))
        return loss
