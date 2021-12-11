# -*- coding: utf-8 -*-
from __future__ import division
import os
import argparse
import tensorflow as tf
from utils.model_utils import make_dir
from dataset.dataset import ImageDataSet
from model.model import yolo_net, yolo_loss


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
GRID_W, GRID_H = 13, 13
N_CLASSES = 8
N_ANCHORS = 5
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH = GRID_H * SCALE, GRID_W * SCALE, 2
batch_size = args.batch_size
data_path = args.data_path

train_dataset = ImageDataSet(data_set='train',
                             mode='train',
                             load_to_memory=False,
                             datapath=data_path)
test_dataset = ImageDataSet(data_set='test',
                            mode='test',
                            flip=False,
                            aug_hsv=False,
                            random_scale=False,
                            load_to_memory=False,
                            datapath=data_path)
num_val_step = int(test_dataset.num_samples / args.batch_size)
save_steps = int(train_dataset.num_samples / args.batch_size * args.save_interval)


def print_info():
    print("train samples: {}".format(train_dataset.num_samples))
    print("test samples: {}".format(test_dataset.num_samples))
    print("batch_size: {}".format(args.batch_size))
    print("iter steps: {}".format(args.num_iter))


def train(load_weights='False'):
    make_dir(args.save_dir)
    max_val_loss = 99999999.0
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.001,
                                               global_step,
                                               1500,
                                               0.96,
                                               staircase=True)

    image = tf.placeholder(
        shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH],
        dtype=tf.float32,
        name='image_placeholder')
    label = tf.placeholder(shape=[None, GRID_H, GRID_W, N_ANCHORS, 8],
                           dtype=tf.float32,
                           name='label_placeholder')
    train_flag = tf.placeholder(dtype=tf.bool, name='flag_placeholder')

    with tf.variable_scope('net'):
        y = yolo_net(image, train_flag)
    with tf.name_scope('loss'):
        loss, loss_xy, loss_wh, loss_re, loss_im, loss_obj, loss_no_obj, loss_c = yolo_loss(
            y, label, batch_size)

    loss_xy_sum = tf.summary.scalar("loss_xy_sum", loss_xy)
    loss_wh_sum = tf.summary.scalar("loss_wh_sum", loss_wh)
    loss_re_sum = tf.summary.scalar("loss_re_sum", loss_re)
    loss_im_sum = tf.summary.scalar("loss_im_sum", loss_im)
    loss_obj_sum = tf.summary.scalar("loss_obj_sum", loss_obj)
    loss_no_obj_sum = tf.summary.scalar("loss_no_obj_sum", loss_no_obj)
    loss_c_sum = tf.summary.scalar("loss_c", loss_c)
    loss_sum = tf.summary.scalar("loss", loss)
    loss_tensorboard_sum = tf.summary.merge([
        loss_xy_sum, loss_wh_sum, loss_re_sum, loss_im_sum,
        loss_obj_sum, loss_no_obj_sum, loss_c_sum, loss_sum
    ])
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = opt.minimize(loss, global_step=global_step)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("./logs", sess.graph)

    if load_weights == 'True':
        print("load weights from {}".format(args.weights_path))
        saver = tf.train.import_meta_graph(args.weights_path + '.meta')
        saver.restore(sess, args.weights_path)
        print('load weights done!')

    for step, (train_image_data, train_label_data) in enumerate(
            train_dataset.get_batch(batch_size)):
        _, lr, train_loss, data, summary_str = sess.run(
            [train_step, learning_rate, loss, y, loss_tensorboard_sum],
            feed_dict={
                train_flag: True,
                image: train_image_data,
                label: train_label_data
            })
        writer.add_summary(summary_str, step)

        if step % 10 == 0:
            print('iter: %i, loss: %f, lr: %f' % (step, train_loss, lr))
        if (step + 1) % save_steps == 0:
            print("val...")
            val_loss = 0.0
            for val_step, (val_image_data, val_label_data) in enumerate(
                    test_dataset.get_batch(batch_size)):
                val_loss += sess.run(loss,
                                     feed_dict={
                                         train_flag: False,
                                         image: val_image_data,
                                         label: val_label_data
                                     })
                if val_step + 1 == num_val_step:
                    break
            val_loss /= num_val_step
            print("iter: {} val_loss: {:.2f}".format(step, val_loss))
            if val_loss < max_val_loss:
                saver.save(sess,
                           os.path.join(
                               args.save_dir,
                               'sgm_2channel416_train_loss_{:.2f}_val_loss_{:.2f}'.format(
                                   train_loss, val_loss)),
                           global_step=global_step)
                max_val_loss = val_loss
        if step + 1 == args.num_iter:
            saver.save(sess,
                       os.path.join(
                           args.save_dir,
                           'sgm_2channel416_final_train_loss_{:.2f}'.format(
                               train_loss)),
                       global_step=global_step)
            print("training done!")
            break


if __name__ == "__main__":
    print_info()
    train(load_weights=args.load_weights)
