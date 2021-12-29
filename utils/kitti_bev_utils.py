import numpy as np
import math
import cv2
import utils.kitti_utils as kitti_utils

def removePoints(PointCloud, BoundaryCond):
    # Boundary condition
    minX = BoundaryCond['minX']
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY']
    maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ']
    maxZ = BoundaryCond['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0] <= maxX) & (PointCloud[:, 1] >= minY) & (
            PointCloud[:, 1] <= maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2] <= maxZ))
    PointCloud = PointCloud[mask]

    PointCloud[:, 2] = PointCloud[:, 2] - minZ

    return PointCloud


def read_labels_for_bevbox(objects):
    bbox_selected = []
    for obj in objects:
        if obj.cls_id != -1:
            bbox = []
            bbox.append(obj.cls_id)
            bbox.extend([obj.t[0], obj.t[1], obj.t[2], obj.h, obj.w, obj.l, obj.ry])
            bbox_selected.append(bbox)
    
    if (len(bbox_selected) == 0):
        return np.zeros((1, 8), dtype=np.float32), True
    else:
        bbox_selected = np.array(bbox_selected).astype(np.float32)
        return bbox_selected, False

# bev image coordinates format
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)

    # front left
    bev_corners[0, 0] = x - w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)
    bev_corners[0, 1] = y - w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)

    # rear left
    bev_corners[1, 0] = x - w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
    bev_corners[1, 1] = y - w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

    # rear right
    bev_corners[2, 0] = x + w / 2 * np.cos(yaw) + l / 2 * np.sin(yaw)
    bev_corners[2, 1] = y + w / 2 * np.sin(yaw) - l / 2 * np.cos(yaw)

    # front right
    bev_corners[3, 0] = x + w / 2 * np.cos(yaw) - l / 2 * np.sin(yaw)
    bev_corners[3, 1] = y + w / 2 * np.sin(yaw) + l / 2 * np.cos(yaw)

    return bev_corners


def inverse_yolo_target(targets, bc):
    ntargets = 0
    for i, t in enumerate(targets):
        if t.sum(0):ntargets += 1
    
    labels = np.zeros([ntargets, 8], dtype=np.float32)

    n = 0
    for t in targets:
        if t.sum(0) == 0:
            continue

        c, y, x, w, l, im, re = t        
        z, h = -1.55, 1.5
        # AVG heights for different classes
        if c == 1: 
            h = 1.8
        elif c == 2:
            h = 4.0
        elif c == 3:
            h = 1.8
        elif c == 4:
            h = 1.2
        elif c == 6:
            h = 1.6
        elif c == 7:
            h = 3.5


        y = y * (bc["maxY"] - bc["minY"]) + bc["minY"]
        x = x * (bc["maxX"] - bc["minX"]) + bc["minX"]
        w = w * (bc["maxY"] - bc["minY"])
        l = l * (bc["maxX"] - bc["minX"])

        #factor = math.atan2(y, x+0.001)
        #y = y * ( 1 + (bc["maxX"] - x)/abs(y)/30)
        #w -= 0.3
        #l -= 0.3

        labels[n, :] = c, x, y, z, h, w, l, - np.arctan2(im, re) - 2*np.pi
        n += 1

    return labels

#send parameters in bev image coordinates format
def drawRotatedBox(img,x,y,w,l,yaw,color):
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2)
    cv2.line(img, (corners_int[0, 0], corners_int[0, 1]), (corners_int[3, 0], corners_int[3, 1]), (255, 255, 0), 2)