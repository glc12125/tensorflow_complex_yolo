# File Name : kitti_aug_utils.py
# Source : https://github.com/jeasinema/VoxelNet-tensorflow/blob/master/utils/utils.py

import numpy as np
import math
import cv2


def lidar_to_camera(x, y, z,V2C=None, R0=None, P2=None):
	p = np.array([x, y, z, 1])
	if V2C is None or R0 is None:
		p = np.matmul(cnf.Tr_velo_to_cam, p)
		p = np.matmul(cnf.R0, p)
	else:
		p = np.matmul(V2C, p)
		p = np.matmul(R0, p)
	p = p[0:3]
	return tuple(p)


def lidar_to_camera_box(boxes,V2C=None, R0=None, P2=None):
	# (N, 7) -> (N, 7) x,y,z,h,w,l,r
	ret = []
	for box in boxes:
		x, y, z, h, w, l, rz = box
		(x, y, z), h, w, l, ry = lidar_to_camera(
			x, y, z,V2C=V2C, R0=R0, P2=P2), h, w, l, -rz - np.pi / 2
		#ry = angle_in_limit(ry)
		ret.append([x, y, z, h, w, l, ry])
	return np.array(ret).reshape(-1, 7)
