import numpy as np
import cv2
import os
from prefetch_generator import BackgroundGenerator
from utils.kitti_utils import read_anchors_file, read_label_from_txt, \
    load_kitti_calib, get_target, removePoints, makeBVFeature
from dataset.augument import RandomScaleAugmentation
from model.model import encode_label
img_h, img_w = 768, 1024
grid_h, grid_w = 24, 32
iou_th = 0.5
boundary = {'minX': 0, 'maxX': 80, 'minY': -40, 'maxY': 40, 'minZ': -2, 'maxZ': 1.25}


class PointCloudDataset(object):

    def __init__(self, root='./kitti/', data_set='train'):

        self.root = root
        self.data_path = os.path.join(root, 'training')
        self.lidar_path = os.path.join(self.data_path, "velodyne/")
        self.calib_path = os.path.join(self.data_path, "calib/")
        self.label_path = os.path.join(self.data_path, "label_2/")
        # self.image_path = os.path.join(self.data_path, "image_2/")

        with open(os.path.join(self.data_path, '%s.txt' % data_set)) as f:
            self.index_list = f.read().splitlines()

    def getitem(self):
        for index in self.index_list:
            lidar_file = self.lidar_path + '/' + index + '.bin'
            calib_file = self.calib_path + '/' + index + '.txt'
            label_file = self.label_path + '/' + index + '.txt'
            calib = load_kitti_calib(calib_file)
            target = get_target(label_file, calib['Tr_velo2cam'])
            # load point cloud data
            point_cloud = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
            b = removePoints(point_cloud, boundary)
            rgb_map = makeBVFeature(b)   # (768, 1024, 3)

            yield index, rgb_map, target


class ImageDataSet(object):

    def __init__(self, data_set='train', mode='train', flip=True, random_scale=True, aug_hsv=False, load_to_memory=False):
        self.mode = mode
        self.flip = flip
        self.aug_hsv = aug_hsv
        self.random_scale = random_scale
        self.anchors_path = 'config/kitti_anchors.txt'
        self.labels_dir = data_set + '/' + 'labels/'
        self.images_dir = data_set + '/' + 'images/'
        self.all_image_index = 'config/' + data_set + '_image_list.txt'
        self.load_to_memory = load_to_memory
        self.anchors = read_anchors_file(self.anchors_path)
        self.rand_scale_transform = RandomScaleAugmentation(img_h, img_w)
        self.label = None
        self.img = None
        self.img_index = None
        self.label_encoded = None

    def horizontal_flip(self, image, target):  # target:class, x,y,w,l,angle
        image = np.flip(image, 1)  # image = image[:, ::-1, :]
        image_w = image.shape[1]
        target[:, 1] = image_w - target[:, 1]
        target[:, 5] = -target[:, 5]
        return image, target

    def augment_hsv(self, img):
        fraction = 0.30  # must be < 1.0
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # hue, sat, val
        s = img_hsv[:, :, 1].astype(np.float32)  # saturation
        v = img_hsv[:, :, 2].astype(np.float32)  # value
        a = (np.random.random() * 2 - 1) * fraction + 1
        b = (np.random.random() * 2 - 1) * fraction + 1
        s *= a
        v *= b
        img_hsv[:, :, 1] = s if a < 1 else s.clip(None, 255)
        img_hsv[:, :, 2] = v if b < 1 else v.clip(None, 255)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return img

    def label_box_center_to_corner(self, label):
        label_ = np.copy(label)
        cx = label_[:, 1]
        cy = label_[:, 2]
        w = label_[:, 3]
        l = label_[:, 4]
        label[:, 1] = cx - w / 2.0
        label[:, 2] = cy - l / 2.0
        label[:, 3] = cx + w / 2.0
        label[:, 4] = cy + l / 2.0
        return label

    def label_box_corner_to_center(self, label):
        cx = (label[:, 1] + label[:, 3]) / 2.0
        cy = (label[:, 2] + label[:, 4]) / 2.0
        w = label[:, 3] - label[:, 1]
        l = label[:, 4] - label[:, 2]
        label[:, 1] = cx
        label[:, 2] = cy
        label[:, 3] = w
        label[:, 4] = l
        return label

    def data_generator(self):
        with open(self.all_image_index, 'r') as f:
            index_list = f.readlines()
            if self.load_to_memory:
                all_images = []
                all_labels = []
                all_index = []
                for index in index_list:
                    index = index.strip()
                    label_file = self.labels_dir + index + '.txt'
                    label = read_label_from_txt(label_file)
                    image_path = self.images_dir + index + '.png'
                    img = cv2.imread(image_path)
                    if img is None:
                        print('failed to load image:' + image_path)
                        continue
                    img = np.flip(img, 2)
                    all_index.append(index)
                    all_images.append(img)
                    all_labels.append(label)
                sample_index = [i for i in range(len(all_index))]
                while True:
                    np.random.shuffle(sample_index)
                    for i in sample_index:
                        self.img_index = np.copy(all_index[i])
                        self.label = np.copy(all_labels[i])
                        self.img = np.copy(all_images[i])
                        if self.aug_hsv:
                            if np.random.random() > 0.5:
                                self.img = self.augment_hsv(self.img)
                        if self.flip:
                            if np.random.random() > 0.5:
                                self.img, self.label = self.horizontal_flip(self.img, self.label)
                        if self.random_scale:
                            self.label = self.label_box_center_to_corner(self.label)
                            self.img, self.label = self.rand_scale_transform(self.img, self.label)
                            self.label = self.label_box_corner_to_center(self.label)
                        self.label_encoded = encode_label(self.label, self.anchors, img_w, img_h, grid_w, grid_h,
                                                          iou_th)
                        if self.mode == 'train':
                            yield self.img_index, self.img / 255.0, self.label_encoded
                        if self.mode == 'infer':
                            yield self.img_index, self.img, self.label



            else:
                while True:
                    np.random.shuffle(index_list)
                    for index in index_list:
                        self.img_index = index.strip()
                        label_file = self.labels_dir + self.img_index + '.txt'
                        self.label = read_label_from_txt(label_file)
                        image_path = self.images_dir + self.img_index + '.png'
                        self.img = cv2.imread(image_path)
                        if self.img is None:
                            print('failed to load image:' + image_path)
                            continue
                        self.img = np.flip(self.img, 2)

                        if self.aug_hsv:
                            if np.random.random() > 0.5:
                                self.img = self.augment_hsv(self.img)
                        if self.flip:
                            if np.random.random() > 0.5:
                                self.img, self.label = self.horizontal_flip(self.img, self.label)
                        if self.random_scale:
                            self.label = self.label_box_center_to_corner(self.label)
                            self.img, self.label = self.rand_scale_transform(self.img, self.label)
                            self.label = self.label_box_corner_to_center(self.label)
                        self.label_encoded = encode_label(self.label, self.anchors, img_w, img_h, grid_w, grid_h, iou_th)
                        if self.mode == 'train':
                            yield self.img_index, self.img / 255.0, self.label_encoded
                        if self.mode == 'infer':
                            yield self.img_index, self.img, self.label

    def get_batch(self, batch_size):
        img_batch = []
        label_batch = []
        i = 0
        for img_idx, img, label_encoded in self.data_generator():
            i += 1
            img_batch.append(img)
            label_batch.append(label_encoded)
            if i % batch_size == 0:
                yield np.array(img_batch), np.array(label_batch)
                i = 0
                img_batch = []
                label_batch = []

    # def pre_get_batch(self, batch_size):
    #     data_batch = BackgroundGenerator(self.get_batch(batch_size), 10)
    #     yield data_batch
