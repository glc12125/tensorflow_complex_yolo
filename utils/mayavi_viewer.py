import numpy as np
import cv2
import utils.kitti_utils_visualisation as kitti_utils

colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0], [255, 255, 0], [255, 0, 255], [100, 255, 100], [255, 100, 255], [100, 100, 100]]

def show_image_with_boxes(img, objects, calib, show3d=False):
    ''' Show image with 2D bounding boxes '''

    img2 = np.copy(img) # for 3d bbox
    for obj in objects:
        if obj.type=='Misc':continue
        #cv2.rectangle(img2, (int(obj.xmin),int(obj.ymin)),
        #    (int(obj.xmax),int(obj.ymax)), (0,255,0), 2)
        box3d_pts_2d, box3d_pts_3d = kitti_utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is not None:
            img2 = kitti_utils.draw_projected_box3d(img2, box3d_pts_2d, colors[obj.cls_id])
    if show3d:
        cv2.imshow("img", img2)
    return img2