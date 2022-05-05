import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
import math

import warnings
warnings.filterwarnings('ignore')

def calc_iou(box1, box2):
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    xx1 = np.maximum(box1[0], box2[0])
    yy1 = np.maximum(box1[1], box2[1])
    xx2 = np.minimum(box1[2], box2[2])
    yy2 = np.minimum(box1[3], box2[3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    ovr = inter / (area1 + area2 - inter)

    return ovr

def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices

    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """

    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                  [0, 1, 0],
                  [-math.sin(ry3d), 0, +math.cos(ry3d)]])

    # 3D bounding box corners
    x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
    y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
    z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)

    # translate
    corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
    corners_2D = p2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]

    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

    verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T

    if return_3d:
        return verts3d, corners_3d
    else:
        return verts3d



def post_3d(det_dir, target_dir, calib_dir='/private/pengliang/KITTI3D/training/calib'):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for i in tqdm(sorted(os.listdir(det_dir))):
        pred = np.loadtxt(os.path.join(det_dir, i), dtype=str).reshape(-1, 16)
        with open(os.path.join(calib_dir, i), encoding='utf-8') as f:
            text = f.readlines()
            P2 = np.array(text[2].split(' ')[1:], dtype=np.float32).reshape(3, 4)

        if pred.shape[0] > 0:
            for j in range(len(pred)):
                bbox2d = pred[j, 4:8].astype(np.float32)

                ####################################################################################
                '''refine x'''
                val1 = pred[j, 8:].astype(np.float32)
                range_x = np.arange(-0.3, 0.301, 0.05)
                tmp_box = []
                for k in range_x:
                    p_2d = project_3d(P2, val1[3]+k, val1[4]-val1[0]/2, val1[5],
                                                    val1[1], val1[0], val1[2], val1[6])

                    iou_2d = np.abs((np.min(p_2d[:, 0]) - bbox2d[0]) - (bbox2d[2] - np.max(p_2d[:, 0])))
                    tmp_box.append(iou_2d)

                good_k = range_x[np.argmin(np.array(tmp_box))]
                pred[j, 11] = pred[j, 11].astype(np.float32) + good_k

                ####################################################################################
                '''refine y'''
                val1 = pred[j, 8:].astype(np.float32)
                range_y = np.arange(-0.3, 0.301, 0.05)
                tmp_box = []
                for k in range_y:
                    p_2d = project_3d(P2, val1[3], val1[4]-val1[0]/2+k, val1[5],
                                                    val1[1], val1[0], val1[2], val1[6])

                    iou_2d = np.abs((np.min(p_2d[:, 1]) - bbox2d[1]) - (bbox2d[3] - np.max(p_2d[:, 1])))
                    tmp_box.append(iou_2d)

                good_k = range_y[np.argmin(np.array(tmp_box))]
                pred[j, 12] = pred[j, 12].astype(np.float32) + good_k

                ####################################################################################
                '''refine h'''
                val1 = pred[j, 8:].astype(np.float32)
                range_h = np.arange(-0.3, 0.301, 0.05)
                tmp_box = []
                for k in range_h:
                    p_2d = project_3d(P2, val1[3], val1[4] - val1[0] / 2, val1[5],
                                                    val1[1], val1[0] + k, val1[2], val1[6])

                    m_2d = np.array([bbox2d[0], np.min(p_2d[:, 1]), bbox2d[2], np.max(p_2d[:, 1])])
                    iou_2d = calc_iou(bbox2d, m_2d)
                    tmp_box.append(iou_2d)

                good_k = range_h[np.argmax(np.array(tmp_box))] + val1[0]
                pred[j, 12] = pred[j, 12].astype(np.float32) - pred[j, 8].astype(np.float32) / 2 + good_k / 2
                pred[j, 8] = good_k

        np.savetxt(os.path.join(target_dir, i), pred, fmt='%s')

