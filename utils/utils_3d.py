'''
 Copyright (C) RoboK Limited - All Rights Reserved
 Unauthorized copying of this file, via any medium is strictly prohibited
 Proprietary and confidential
 Written by Liangchuan Gu<liangchuan.gu@robok.ai>, August 2019
'''

from __future__ import division
import numpy as np
import cv2
import math
import os
import six
import utils.kitti_aug_utils as aug_utils
import utils.kitti_bev_utils as bev_utils
import utils.kitti_utils_visualisation as kitti_utils
import utils.mayavi_viewer as mview

if six.PY3:
  buffer_ = memoryview
else:
  buffer_ = buffer  # noqa

def softmax(x):
    #print('In softmax, type(x: {}, )x: {}'.format(type(x), x))
    e_x = np.exp(x)
    return e_x / np.sum(e_x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea

def read_anchors_from_file(file_path):
    """
    Read  anchors from the configuration file
    """
    anchors = []
    with open(file_path, 'r') as file:
        for line in file.read().splitlines():
            anchors.append(list(map(float, line.split())))
    return np.array(anchors)


def read_class(file_path):
    """
    Read class flags for visualization
    """
    classes, names, colors = [], [], []
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
        for line in lines:
            cls, name, color = line.split()
            classes.append(int(cls))
            names.append(name)
            colors.append(eval(color))
    return classes, names, colors


def draw_rotated_box(img, cy, cx, w, h, angle, color):
    """
    param: img(array): RGB image
    param: cy(int, float):  Here cy is cx in the image coordinate system
    param: cx(int, float):  Here cx is cy in the image coordinate system
    param: w(int, float):   box's width
    param: h(int, float):   box's height
    param: angle(float): rz
    param: color(tuple, list): the color of box, (R, G, B)
    """
    #print("cy: {0}, cx: {1}, w: {2}, h: {3}, angle: {4}, color: {5}".format(cy, cx, w, h, angle, color))
    left = int(cy - w / 2)
    top = int(cx - h / 2)
    right = int(cx + h / 2)
    bottom = int(cy + h / 2)
    ro = math.sqrt(pow(left - cy, 2) + pow(top - cx, 2))
    if h == 0:
        a1 = np.arctan(np.inf)
        a2 = -np.arctan(np.inf)
    else:
        a1 = np.arctan((w / 2) / (h / 2))
        a2 = -np.arctan((w / 2) / (h / 2))
    a3 = -np.pi + a1
    a4 = np.pi - a1
    rotated_p1_y = cy + int(ro * np.sin(angle + a1))
    rotated_p1_x = cx + int(ro * np.cos(angle + a1))
    rotated_p2_y = cy + int(ro * np.sin(angle + a2))
    rotated_p2_x = cx + int(ro * np.cos(angle + a2))
    rotated_p3_y = cy + int(ro * np.sin(angle + a3))
    rotated_p3_x = cx + int(ro * np.cos(angle + a3))
    rotated_p4_y = cy + int(ro * np.sin(angle + a4))
    rotated_p4_x = cx + int(ro * np.cos(angle + a4))
    center_p1p2y = int((rotated_p1_y + rotated_p2_y) * 0.5)
    center_p1p2x = int((rotated_p1_x + rotated_p2_x) * 0.5)
    cv2.line(img, (rotated_p1_y, rotated_p1_x), (rotated_p2_y, rotated_p2_x),
             color, 1)
    cv2.line(img, (rotated_p2_y, rotated_p2_x), (rotated_p3_y, rotated_p3_x),
             color, 1)
    cv2.line(img, (rotated_p3_y, rotated_p3_x), (rotated_p4_y, rotated_p4_x),
             color, 1)
    cv2.line(img, (rotated_p4_y, rotated_p4_x), (rotated_p1_y, rotated_p1_x),
             color, 1)
    cv2.line(img, (center_p1p2y, center_p1p2x), (cy, cx), color, 1)

def calculate_angle(im, re):
    """
    param: im(float): imaginary parts of the plural
    param: re(float): real parts of the plural
    return: The angle at which the objects rotate
    around the Z axis in the velodyne coordinate system
    """
    if re > 0:
        return np.arctan(im / re)
    elif im < 0:
        return -np.pi + np.arctan(im / re)
    else:
        return np.pi + np.arctan(im / re)

def process_output_data(data, anchors, important_classes, grid_w, grid_h, net_scale):
    """
    Decode the data output by the model, obtain the center coordinates
    x, y and width and height of the bounding box in the image,
    and the category, the real and imaginary parts of the complex.
    """
    locations = []
    classes = []
    n_anchors = np.shape(anchors)[0]
    #print('n_anchors: {}'.format(n_anchors))
    for i in range(grid_h):
        for j in range(grid_w):
            for k in range(n_anchors):
                class_vec = softmax(data[0, i, j, k, 7:])
                object_conf = sigmoid(data[0, i, j, k, 6])
                class_prob = object_conf * class_vec
                w = np.exp(data[0, i, j, k, 2]
                           ) * anchors[k][0] / 50 * grid_w * net_scale
                h = np.exp(data[0, i, j, k, 3]
                           ) * anchors[k][1] / 50 * grid_h * net_scale
                dx = sigmoid(data[0, i, j, k, 0])
                dy = sigmoid(data[0, i, j, k, 1])
                re = 2 * sigmoid(data[0, i, j, k, 4]) - 1
                im = 2 * sigmoid(data[0, i, j, k, 5]) - 1
                y = (i + dy) * net_scale
                x = (j + dx) * net_scale
                classes.append(class_prob[important_classes])
                locations.append([x, y, w, h, re, im])
    #print('Finished generating prediction results from NN output')
    classes = np.array(classes)
    locations = np.array(locations)
    return classes, locations

def non_max_supression(classes, locations, prob_th, iou_th):
    """
    Filter out some overlapping boxes by non-maximum suppression
    """
    classes = np.transpose(classes)
    indxs = np.argsort(-classes, axis=1)
    #print('in non_max_supression, \nclasses: {}, \nindxs: {}'.format(classes, indxs))
    for i in range(classes.shape[0]):
        classes[i] = classes[i][indxs[i]]

    for class_idx, class_vec in enumerate(classes):
        for roi_idx, roi_prob in enumerate(class_vec):
            if roi_prob < prob_th:
                classes[class_idx][roi_idx] = 0

    for class_idx, class_vec in enumerate(classes):
        for roi_idx, roi_prob in enumerate(class_vec):
            if roi_prob == 0:
                continue
            roi = locations[indxs[class_idx][roi_idx]][0:4]
            for roi_ref_idx, roi_ref_prob in enumerate(class_vec):
                if roi_ref_prob == 0 or roi_ref_idx <= roi_idx:
                    continue
                roi_ref = locations[indxs[class_idx][roi_ref_idx]][0:4]
                if bbox_iou(roi, roi_ref, False) > iou_th:
                    classes[class_idx][roi_ref_idx] = 0
    return classes, indxs

def filter_bbox(classes, rois, indxs):
    """
    Pick out bounding boxes that are retained after non-maximum suppression
    """
    all_bboxs = []
    for class_idx, c in enumerate(classes):
        for loc_idx, class_prob in enumerate(c):
            if class_prob > 0:
                x = int(rois[indxs[class_idx][loc_idx]][0])
                y = int(rois[indxs[class_idx][loc_idx]][1])
                w = int(rois[indxs[class_idx][loc_idx]][2])
                h = int(rois[indxs[class_idx][loc_idx]][3])
                re = rois[indxs[class_idx][loc_idx]][4]
                im = rois[indxs[class_idx][loc_idx]][5]
                all_bboxs.append([class_idx, x, y, w, h, re, im, class_prob])
    return all_bboxs

def get_bbox_corners(box):
    """
    param: box(tuple, list): cx, cy, w, l
    return: (tuple): x_min, y_min, x_max, y_max
    """
    bx = box[0]
    by = box[1]
    bw = box[2]
    bl = box[3]
    top = int((by - bl / 2.0))
    left = int((bx - bw / 2.0))
    right = int((bx + bw / 2.0))
    bottom = int((by + bl / 2.0))
    return left, top, right, bottom

def remove_points(point_cloud, boundary_condition):
    """
    param point_cloud(array): Original point cloud data
    param boundary_condition(dict): The boundary of the area of interest
    return (array): Point cloud data within the area of interest
    """
    # Boundary condition
    min_x = boundary_condition['minX']
    max_x = boundary_condition['maxX']
    min_y = boundary_condition['minY']
    max_y = boundary_condition['maxY']
    min_z = boundary_condition['minZ']
    max_z = boundary_condition['maxZ']
    # Remove the point out of range x,y,z
    mask = np.where((point_cloud[:, 0] >= min_x) & (point_cloud[:, 0] <= max_x)
                    & (point_cloud[:, 1] >= min_y) & (point_cloud[:, 1] <= max_y)
                    & (point_cloud[:, 2] >= min_z) & (point_cloud[:, 2] <= max_z))
    point_cloud = point_cloud[mask]
    point_cloud[:, 2] = point_cloud[:, 2] - min_z
    return point_cloud

IMG_HEIGHT = 416
IMG_WIDTH = 416

def make_birdseye_view_feature(point_cloud_):
    """
    param point_cloud_ (array): Point cloud data within the area of interest
    return (array): RGB map
    """
    # 416 x 416 x 2
    Height = IMG_HEIGHT + 1
    Width = IMG_WIDTH + 1
    half_width = IMG_WIDTH / 2.0
    # Discretize Feature Map
    point_cloud = np.copy(point_cloud_)
    point_cloud[:, 0] = np.int_(np.floor(point_cloud[:, 0] / 50.0 * IMG_HEIGHT))
    point_cloud[:, 1] = np.int_(
        np.floor(point_cloud[:, 1] / 50.0 * IMG_WIDTH) + half_width)
    # sort-3times
    indices = np.lexsort(
        (-point_cloud[:, 2], point_cloud[:, 1], point_cloud[:, 0]))

    point_cloud = point_cloud[indices]
    # Height Map
    height_map = np.zeros((Height, Width))
    _, indices = np.unique(point_cloud[:, 0:2], axis=0, return_index=True)
    point_cloud_frac = point_cloud[indices]
    # some important problem is image coordinate is (y,x), not (x,y)

    height_map[np.int_(point_cloud_frac[:, 0]),
               np.int_(point_cloud_frac[:, 1])] = point_cloud_frac[:, 2]
    # Intensity Map & DensityMap
    #intensity_map = np.zeros((Height, Width))
    density_map = np.zeros((Height, Width))
    _, indices, counts = np.unique(point_cloud[:, 0:2],
                                   axis=0,
                                   return_index=True,
                                   return_counts=True)
    point_cloud_top = point_cloud[indices]
    normalized_counts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    #intensity_map[np.int_(point_cloud_top[:, 0]),
    #              np.int_(point_cloud_top[:, 1])] = point_cloud_top[:, 3]
    density_map[np.int_(point_cloud_top[:, 0]),
                np.int_(point_cloud_top[:, 1])] = normalized_counts

    rgb_map = np.zeros((IMG_HEIGHT, IMG_WIDTH, 2))
    rgb_map[:, :, 0] = density_map[:IMG_HEIGHT, :IMG_WIDTH]  # r_map
    rgb_map[:, :, 1] = height_map[:IMG_HEIGHT, :IMG_WIDTH] / 3.26  # g_map
    #rgb_map[:, :, 2] = intensity_map  # b_map

    return rgb_map

def rescale_boxes(boxes, current_dims, original_dims):
    print(boxes)
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_dims
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dims[0] / max(original_dims))
    pad_y = max(orig_w - orig_h, 0) * (current_dims[1] / max(original_dims))

    # Image height and width after padding is removed
    unpad_h = current_dims[0] #- pad_y
    unpad_w = current_dims[1] #- pad_x
    # Rescale bounding boxes to dimension of original image
    temp_boxes = np.array(boxes)
    temp_boxes[:, 1] = ((temp_boxes[:, 1] - pad_x // 2) / unpad_w) * orig_w
    temp_boxes[:, 2] = ((temp_boxes[:, 2] - pad_y // 2) / unpad_h) * orig_h
    temp_boxes[:, 3] = ((temp_boxes[:, 3] - pad_x // 2) / unpad_w) * orig_w
    temp_boxes[:, 4] = ((temp_boxes[:, 4] - pad_y // 2) / unpad_h) * orig_h

    return temp_boxes.tolist()

def predictions_to_kitti_format(img_detections, calib, img_shape_2d, img_width, img_height):

    predictions = np.zeros([50, 7], dtype=np.float32)
    count = 0
    #print(img_detections)
    for detections in img_detections:
        if detections is None:
            continue
        #print("detections: {}".format(detections))
        # Rescale boxes to original image
        x = detections[1]
        y = detections[2]
        w = detections[3]
        l = detections[4]
        re = detections[5]
        im = detections[6]
        cls_pred = detections[0]
        #yaw = np.arctan2(im, re)
        predictions[count, :] = cls_pred, x/img_width, y/img_width, w/img_width, l/img_width, im, re
        count += 1

    boundary = {
            "minX": 0,
            "maxX": 50,
            "minY": -25,
            "maxY": 25,
            "minZ": -2.73,
            "maxZ": 1.27
        }
    #print("detections before inverse_yolo_target: {}".format(predictions))
    predictions = bev_utils.inverse_yolo_target(predictions, boundary)
    #print("detections after inverse_yolo_target(before lidar_to_camera_box): {}".format(predictions))
    #print(predictions)
    #print(predictions.shape[0])
    if predictions.shape[0]:
        predictions[:, 1:] = aug_utils.lidar_to_camera_box(predictions[:, 1:], calib.V2C, calib.R0, calib.P)
    #print("detections after lidar_to_camera_box: {}".format(predictions))
    objects_new = []
    corners3d = []
    for index, l in enumerate(predictions):

        str = "Misc"
        if l[0] == 0: str="Car"
        elif l[0] == 1: str="Van"
        elif l[0] == 2: str="Truck"
        elif l[0] == 3: str="Pedestrian"
        elif l[0] == 4: str="Person_sitting"
        elif l[0] == 5: str="Cyclist"
        elif l[0] == 6: str="Tram"
        else: str = "Misc"
        line = '%s -1 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0' % str

        obj = kitti_utils.Object3d(line)
        obj.t = l[1:4]
        obj.h,obj.w,obj.l = l[4:7]
        obj.ry = np.arctan2(math.sin(l[7]), math.cos(l[7]))

        _, corners_3d = kitti_utils.compute_box_3d(obj, calib.P)
        corners3d.append(corners_3d)
        objects_new.append(obj)

    #print(corners3d)
    if len(corners3d) > 0:
        corners3d = np.array(corners3d)
        img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

        img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape_2d[1] - 1)
        img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape_2d[0] - 1)
        img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape_2d[1] - 1)
        img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape_2d[0] - 1)

        img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
        img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
        box_valid_mask = np.logical_and(img_boxes_w < img_shape_2d[1] * 0.8, img_boxes_h < img_shape_2d[0] * 0.8)

    for i, obj in enumerate(objects_new):
        x, z, ry = obj.t[0], obj.t[2], obj.ry
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry

        obj.alpha = alpha
        obj.box2d = img_boxes[i, :]

    return objects_new

class PointCloudDataset(object):
    def __init__(self,
                 root='./tests/'):
        self.root = root
        self.pointcloud_path = os.path.join(root, 'pointcloud')
        self.boundary = {
            "minX": 0,
            "maxX": 50,
            "minY": -25,
            "maxY": 25,
            "minZ": -2.73,
            "maxZ": 1.27
        }

    def getitem(self, name):
        """
        Encode single-frame point cloud data into RGB-map and get the label
        """
        pointcloud_file = self.pointcloud_path + '/' + name + '.bin'

        # load point cloud data
        point_cloud = np.fromfile(pointcloud_file,
                                  dtype=np.float32).reshape(-1, 4)

        b = remove_points(point_cloud, self.boundary)
        rgb_map = make_birdseye_view_feature(b)  # (416, 416, 2)
        return rgb_map

class PointCloudProcessor(object):
    def __init__(self,
                 root='./tests/'):
        self.root = root
        self.boundary = {
            "minX": 0,
            "maxX": 50,
            "minY": -25,
            "maxY": 25,
            "minZ": -2.73,
            "maxZ": 1.27
        }

    def getPointCloud(self, pointcloudData, dimension = 3):
        """
        Encode single-frame point cloud data into RGB-map and get the label
        """
        #print('Convert to buffer...')
        pointcloudBuffer = buffer_(pointcloudData)

        print('Reshape...')
        point_cloud = np.frombuffer(pointcloudBuffer, dtype=np.float32).reshape(-1, dimension)
        np.savetxt('point_cloud_from_tlm_without_remove.txt', point_cloud, delimiter=',', fmt='%.5f')
        #print('Transforming...')
        # Hard code for testing
        calib = kitti_utils.Calibration('tests/calib/000248.txt')
        point_cloud = calib.project_image_to_velo(point_cloud)

        print("remove_points")
        b = remove_points(point_cloud, self.boundary)

        np.savetxt('point_cloud_from_tlm.txt', b, delimiter=',', fmt='%.5f')

        print("make_birdseye_view_feature")
        rgb_map = make_birdseye_view_feature(b)  # (416, 416, 2)

        return rgb_map