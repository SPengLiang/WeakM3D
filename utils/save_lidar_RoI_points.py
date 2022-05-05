from sklearn.cluster import DBSCAN
import os
from tqdm import tqdm
import numpy as np
import cv2 as cv
import pickle

import sys
sys.path.append(os.getcwd())
from dataloader import calib_parse


kitti_raw_file_name = './data/kitti/data_file/split/train_raw.txt'


def obtain_cluster_RoI_points_for_raw_kitti(l_3d, l_2d, seg_bbox_path, seg_mask_path, file_ind, P2):
    bbox2d = np.loadtxt(seg_bbox_path).reshape(-1, 5)
    bbox_mask = (np.load(seg_mask_path))['masks']

    all_box, all_RoI_points = [], []
    for index, b in enumerate(bbox2d):
        if b[-1] < 0.7:
            continue

        obj_mask = bbox_mask[index]
        obj_mask[obj_mask < 0.7] = 0
        obj_mask[obj_mask >= 0.7] = 1
        ind = obj_mask[l_2d[:, 1], l_2d[:, 0]].astype(np.bool)

        cam_points = l_3d[ind]
        if len(cam_points) < 10:
            continue

        cluster_index = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1).fit_predict(cam_points)

        cam_points = cam_points[cluster_index > -1]
        cluster_index = cluster_index[cluster_index > -1]

        if len(cam_points) < 10:
            continue

        cluster_set = set(cluster_index[cluster_index > -1])
        cluster_sum = np.array([len(cam_points[cluster_index == i]) for i in cluster_set])
        RoI_points = cam_points[cluster_index == np.argmax(cluster_sum)]

        all_box.append(b[:-1])
        all_RoI_points.append(RoI_points)

    return np.array(all_box), np.array(all_RoI_points)


if __name__ == '__main__':
    train_file = np.loadtxt(kitti_raw_file_name, dtype=str)
    img_list, velo_list, calib_cam_to_cam_list, calib_velo_to_cam_list = \
        train_file[:, 0], train_file[:, 1], train_file[:, 2], train_file[:, 3]

    for i, _ in enumerate(tqdm(img_list)):
        raw_lidar_path = velo_list[i]

        seg_mask_path = img_list[i].replace('image_02', 'seg_mask').replace('png', 'npz')
        seg_bbox_path = img_list[i].replace('image_02', 'seg_bbox').replace('png', 'txt')
        lidar_save_RoI_points_path = img_list[i].replace('image_02', 'lidar_RoI_points').replace('png', 'pkl')

        lidar_save_RoI_points_dir = os.path.dirname(lidar_save_RoI_points_path)
        if not os.path.exists(lidar_save_RoI_points_dir):
            os.makedirs(lidar_save_RoI_points_dir)

        l_3d = np.fromfile(str(raw_lidar_path), dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]

        # obtain 2d coordinates
        calib = calib_parse.parse_calib('raw', [calib_cam_to_cam_list[i], calib_velo_to_cam_list[i]])
        l_3d = (calib['l2i'] @ np.concatenate([l_3d, np.ones_like(l_3d[:, 0:1])], axis=1).T).T
        l_2d = (calib['P2'] @ np.concatenate([l_3d, np.ones_like(l_3d[:, 0:1])], axis=1).T).T
        l_2d = (l_2d[:, :2] / l_2d[:, 2:3]).astype(np.int)

        # remove points outside fov
        rgb_img = cv.imread(img_list[i])
        h, w, _ = rgb_img.shape
        valid_ind = (l_2d[:, 0] > 0) & (l_2d[:, 0] < w) & (l_2d[:, 1] > 0) & (l_2d[:, 1] < h) & (l_3d[:, 2] > 0)
        l_3d, l_2d = l_3d[valid_ind], l_2d[valid_ind]

        RoI_box_points = {'bbox2d':[], 'RoI_points':[]}
        if not os.path.exists(seg_bbox_path):
            with open(lidar_save_RoI_points_path, 'wb') as f:
                pickle.dump(RoI_box_points, f)
            continue

        bbox2d = np.loadtxt(seg_bbox_path).reshape(-1, 5)
        if len(bbox2d) < 1:
            with open(lidar_save_RoI_points_path, 'wb') as f:
                pickle.dump(RoI_box_points, f)
            continue

        cls_bbox2d, RoI_points = obtain_cluster_RoI_points_for_raw_kitti(
                    l_3d, l_2d, seg_bbox_path, seg_mask_path, i, calib['P2'])

        RoI_box_points['bbox2d'] = cls_bbox2d
        RoI_box_points['RoI_points'] = RoI_points
        with open(lidar_save_RoI_points_path, 'wb') as f:
            pickle.dump(RoI_box_points, f)