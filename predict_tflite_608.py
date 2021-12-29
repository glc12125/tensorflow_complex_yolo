# -*- coding: utf-8 -*-
from __future__ import division
import cv2
import copy
import argparse
import numpy as np
import tensorflow as tf
from model.model_608 import YoloLoss
from dataset.dataset_608 import PointCloudDataset
from utils.model_utils_608 import preprocess_data, non_max_supression, filter_bbox, make_dir
from utils.kitti_utils_608 import draw_rotated_box, calculate_angle, get_corner_gtbox, \
    read_anchors_from_file, read_class_flag

from utils.utils_3d import draw_rotated_box, calculate_angle, \
    read_anchors_from_file, read_class, process_output_data, \
    non_max_supression, filter_bbox, get_bbox_corners, PointCloudProcessor, predictions_to_kitti_format, \
    rescale_boxes
import utils.kitti_utils_visualisation as kitti_utils
import utils.mayavi_viewer as mview

gt_box_color = (255, 255, 255)
prob_th = 0.3
nms_iou_th = 0.4
n_anchors = 5
n_classes = 8
net_scale = 32
img_h, img_w = 608, 608
grid_w, grid_h = 19, 19
class_list = [
    'Car', 'Van', 'Truck', 'Pedestrian',
    'Person_sitting', 'Cyclist', 'Tram', 'Misc'
]


parser = argparse.ArgumentParser()
parser.add_argument("--draw_gt_box", type=str,  default='True', help="Whether to draw_gtbox, True or False")
parser.add_argument("--weights_path", type=str, default='./weights/yolo_tloss_1.185166835784912_vloss_2.9397876932621-220800',
                    help="set the weights_path")
parser.add_argument("--save_path", type=str, default='./save_path',
                    help="set the path to save the predictions")
args = parser.parse_args()
weights_path = args.weights_path
save_path = args.save_path
# dataset
dataset = PointCloudDataset(root='./kitti/', data_set='test')
make_dir(save_path)

def evaluate_tflite_model(tflite_save_path, x_test, batch_size=8):
    """Calculate the accuracy of a TensorFlow Lite model using TensorFlow Lite interpreter.

    Args:
        tflite_save_path: Path to TensorFlow Lite model to test.
        x_test: numpy array of testing data.
    """

    interpreter = tf.lite.Interpreter(model_path=str(tflite_save_path))

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    accuracy_count = 0
    num_test_images = len(y_test)
    loss = 0
    yolo_loss = YoloLoss(batch_size=batch_size)
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

def visualize_in_image(predictions, img_path, bev_width, bev_height, bev_img, img_idx):
    img2d = cv2.imread(img_path)
    calib = kitti_utils.Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
    objects_pred = predictions_to_kitti_format(predictions, calib, img2d.shape, bev_width, bev_height)
    img2d = mview.show_image_with_boxes(img2d, objects_pred, calib, False)
    detected_img = copy.deepcopy(img2d)
    r_channel_place_holder = np.zeros((bev_height, bev_width))
    bev_img = np.dstack((bev_img, r_channel_place_holder))
    #cv2.imshow("birdseye view of point clouds", bev_img)
    #img2d = cv2.copyMakeBorder(img2d, 0, bev_img.shape[0] - img2d.shape[0], 0, 0, cv2.BORDER_CONSTANT,value=[255,255,255])
    #numpy_vertical_concat = np.concatenate((bev_img, img2d), axis=1)
    #cv2.imshow("3d bounding boxes on image for pointcloud", img2d)
    cv2.imwrite('{}/{}.png'.format("rendered_front_view_tflite", img_idx), img2d)
    cv2.waitKey(0)

    return detected_img

def predict(draw_gt_box='False'):

    important_classes, names, color = read_class_flag('config/class_flag.txt')
    anchors = read_anchors_from_file('config/kitti_anchors.txt')

    interpreter = tf.lite.Interpreter(model_path=str(weights_path))

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #sess = tf.compat.v1.Session()
    #saver = tf.train.import_meta_graph(weights_path + '.meta')
    #saver.restore(sess, weights_path)
    #graph = tf.get_default_graph()
    #image = graph.get_tensor_by_name("input_1:0")
    #train_flag = graph.get_tensor_by_name("flag_placeholder:0")
    #y = graph.get_tensor_by_name("net/y:0")
    for img_idx, rgb_map, target in dataset.getitem():
        print("process data: {}, saved in {}/".format(img_idx, save_path))
        img = np.array(rgb_map * 255, np.uint8)
        target = np.array(target)
        # draw gt bbox
        if draw_gt_box == 'True':
            for i in range(target.shape[0]):
                if target[i].sum() == 0:
                    break
                cx = int(target[i][1] * img_w)
                cy = int(target[i][2] * img_h)
                w = int(target[i][3] * img_w)
                h = int(target[i][4] * img_h)
                rz = target[i][5]
                draw_rotated_box(img, cx, cy, w, h, rz, gt_box_color)
                label = class_list[int(target[i][0])]
                box = get_corner_gtbox([cx, cy, w, h])
                print("GT: {}".format(box))
                cv2.putText(img, label, (box[0], box[1]),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, gt_box_color, 1)
        test_input = np.asarray(rgb_map)
        #data = new_model.predict(test_input.astype(np.float32))
        test_input = test_input.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input[np.newaxis, ...])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.squeeze(output_data)
        pred = tf.reshape(output_data, shape=(-1, grid_h, grid_w, n_anchors, 7 + n_classes))
        #data = sess.run(y, feed_dict={image: [rgb_map], train_flag: False})
        classes, rois = preprocess_data(pred, anchors, important_classes,
                                        grid_w, grid_h, net_scale)
        classes, index = non_max_supression(classes, rois, prob_th, nms_iou_th)
        all_boxes = filter_bbox(classes, rois, index)
        for box in all_boxes:
            print("Prediction: {}".format(box))
            class_idx = box[0]
            corner_box = get_corner_gtbox(box[1:5])
            angle = calculate_angle(box[6], box[5])
            class_prob = box[7]
            if box[3] > img_w:
                continue
            else:
                w = box[3]
            if box[4] > img_h:
                continue
            else:
                h = box[4]
            draw_rotated_box(img, box[1], box[2], w, h,
                             angle, color[class_idx])
            cv2.putText(img,
                        class_list[class_idx] + ' : {:.2f}'.format(class_prob),
                        (corner_box[0], corner_box[1]), cv2.FONT_HERSHEY_PLAIN,
                        0.7, color[class_idx], 1, cv2.LINE_AA)
        cv2.imwrite('{}/{}.png'.format(save_path, img_idx), img)
        print('Showing 3d bounding boxes in image...')
        img_path = "kitti/training/image_2/" + str(img_idx) + ".png"
        resultImg2D = visualize_in_image(all_boxes, img_path, img_w, img_h, img, img_idx)

if __name__ == '__main__':
    predict(draw_gt_box=args.draw_gt_box)
