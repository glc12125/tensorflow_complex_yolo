import numpy as np


def iou_wh(box1_wh, box2_wh):
    """
    param box1_wh (list, tuple): Width and height of a box
    param box2_wh (list, tuple): Width and height of a box
    return (float): iou
    """
    min_w = min(box1_wh[0], box2_wh[0])
    min_h = min(box1_wh[1], box2_wh[1])
    area_r1 = box1_wh[0] * box1_wh[1]
    area_r2 = box2_wh[0] * box2_wh[1]
    intersect = min_w * min_h
    union = area_r1 + area_r2 - intersect
    return intersect / union


def get_grid_cell(roi, img_w, img_h, grid_w, grid_h):  # roi[x, y, w, h, rz]
    """
    Get the grid cell into which the object falls
    param roi : [x, y, w, h, rz]
    param img_w: The width of images
    param img_h: The height of images
    param grid_w: 
    param grid_h:
    return (int, int):
    """
    x_center = roi[0]
    y_center = roi[1]
    grid_x = np.minimum(int(grid_w * x_center / img_w), grid_w-1)
    grid_y = np.minimum(int(grid_h * y_center / img_h), grid_h-1)
    return grid_x, grid_y

def get_active_anchors(box_w_h, anchors, iou_th):
    """
    Get the index of the anchor that matches the ground truth box
    param box_w_h (list, tuple):  Width and height of a box
    param anchors (array): anchors
    param iou_th: Match threshold
    return (list):
    """
    index = []
    iou_max, index_max = 0, 0
    for i, a in enumerate(anchors):
        iou = iou_wh(box_w_h, a)
        if iou > iou_th:
            index.append(i)
        if iou > iou_max:
            iou_max, index_max = iou, i
    if len(index) == 0:
        index.append(index_max)
    return index

def roi2label(roi, anchor, img_w, img_h, grid_w, grid_h):
    """
    Encode the label to match the model output format
    param roi: x, y, w, h, angle
    
    return: encoded label 
    """
    x_center = roi[0]
    y_center = roi[1]
    w = grid_w * roi[2] / img_w
    h = grid_h * roi[3] / img_h
    anchor_w = grid_w * anchor[0] / img_w
    anchor_h = grid_h * anchor[1] / img_h
    grid_x = grid_w * x_center / img_w
    grid_y = grid_h * y_center / img_h
    grid_x_offset = grid_x - int(grid_x)
    grid_y_offset = grid_y - int(grid_y)
    roi_w_scale = np.log(w / anchor_w + 1e-16)
    roi_h_scale = np.log(h / anchor_h + 1e-16)
    re = np.cos(roi[4])
    im = np.sin(roi[4])
    label = [grid_x_offset, grid_y_offset, roi_w_scale, roi_h_scale, re, im]
    return label

def encode_label(labels, anchors, img_w, img_h, grid_w, grid_h, iou_th):
    """
    Encode the label to match the model output format
    param labels (array): x, y, w, h, angle
    param anchors (array): anchors
    return: encoded label 
    """
    anchors_on_image = np.array([img_w, img_h]) * anchors / np.array([50, 50])
    n_anchors = np.shape(anchors_on_image)[0]
    label_encoded = np.zeros([grid_h, grid_w, n_anchors, (6 + 1 + 1)],
                             dtype=np.float32)
    if labels is None:
        print("labels are None!!!!!")
        return label_encoded
    for i in range(labels.shape[0]):
        rois = labels[i][1:]
        classes = np.array(labels[i][0], dtype=np.int32)
        active_indexes = get_active_anchors(rois[2:4], anchors_on_image, iou_th)
        grid_x, grid_y = get_grid_cell(rois, img_w, img_h, grid_w, grid_h)
        for active_index in active_indexes:
            anchor_label = roi2label(rois, anchors_on_image[active_index],
                                     img_w, img_h, grid_w, grid_h)
            label_encoded[grid_y, grid_x, active_index] = np.concatenate(
                (anchor_label, [classes], [1.0]))
    return label_encoded