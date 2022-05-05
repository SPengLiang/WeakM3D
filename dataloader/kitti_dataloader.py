import numpy as np
import cv2 as cv
import os
import random
import warnings
import pickle
from tqdm import tqdm
import torch
import torch.utils.data as data
import math
import torch.nn.functional as F
import pickle
from dataloader import calib_parse
from PIL import Image


class KITTI3D_Object_Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.root_3d_path = cfg.DATA.ROOT_3D_PATH
        self.cls_list = cfg.DATA.CLS_LIST
        self.imagenet_stats = {'mean': cfg.DATA.IMAGENET_STATS_MEAN,
                               'std': cfg.DATA.IMAGENET_STATS_STD}


    def pre_process_img(self, img):
        process_img = []
        for j in range(3):
            process_img.append(
                (img[:, :, j] - self.imagenet_stats['mean'][j]) / self.imagenet_stats['std'][j])
        process_img = np.stack(process_img, axis=2)
        return process_img

    def deprocess(self, input):
        for j in range(3):
            input[:, :, j] = input[:, :, j] * self.imagenet_stats['std'][j] + self.imagenet_stats['mean'][j]
        input *= 255
        return input

    def load_rgb_image(self, path):
        cv.setNumThreads(0)

        img = cv.imread(path) / 255.
        return self.pre_process_img(img)

    def constrain_box(self, org_box, hw):
        box = org_box.copy()
        if box[2] <= 0 or box[0] >= hw[1]-1 or box[1] >= hw[0]-1 or box[3] <= 0:
            return None
        box[0] = max(box[0], 0)
        box[1] = max(box[1], 0)
        box[2] = min(box[2], hw[1]-1)
        box[3] = min(box[3], hw[0]-1)
        return box

    def cut_or_pad_img(self, img_path, targetHW):
        # print(img_path)
        cv.setNumThreads(0)

        img = cv.imread(img_path) / 255.
        bbox_shift = np.array([[0, 0, 0, 0]])

        t_H, t_W = targetHW
        H, W = img.shape[0], img.shape[1]

        padW = np.abs(t_W - W)
        half_padW = int(padW//2)
        # crop
        if W > t_W:
            img = img[:, half_padW:half_padW+t_W]
            bbox_shift[0, [0, 2]] -= half_padW
        # pad
        elif W < t_W:
            img = np.pad(img, [(0, 0), (half_padW, padW-half_padW), (0, 0)], 'constant')
            bbox_shift[0, [0, 2]] += half_padW

        # crop
        padH = np.abs(t_H - H)
        if H > t_H:
            img = img[padH:, :]
            bbox_shift[0, [1, 3]] -= padH
        # pad
        elif H < t_H:
            padH = t_H - H
            img = np.pad(img, [(padH, 0), (0, 0), (0, 0)], 'constant')
            bbox_shift[0, [1, 3]] += padH

        img = self.pre_process_img(img)

        return img, bbox_shift


    def __getitem__(self, index):
        pass

    def __len__(self):
        pass



class KITTI3D_Object_Dataset_Raw_RoI_Lidar(KITTI3D_Object_Dataset):
    def __init__(self, cfg):
        super(KITTI3D_Object_Dataset_Raw_RoI_Lidar, self).__init__(cfg)
        self.train_file = np.loadtxt(cfg.TRAIN.TRAIN_FILE, dtype=str)

        self.data_mode = cfg.DATA.MODE
        if self.data_mode == 'KITTI 3D':
            self.img_list = np.array([os.path.join(cfg.DATA.ROOT_3D_PATH, 'image_2', i+'.png') for i in self.train_file])
            self.calib_list = np.array([os.path.join(cfg.DATA.ROOT_3D_PATH, 'calib', i+'.txt') for i in self.train_file])
            self.lidar_RoI_points_path_list = [i.replace('image_2', cfg.DATA.RoI_POINTS_DIR).replace('png', 'pkl')
                                               for i in self.img_list]
        else:
            self.img_list, self.velo_list, self.calib_cam_to_cam_list, self.calib_velo_to_cam_list = \
                self.train_file[:, 0], self.train_file[:, 1], self.train_file[:, 2], self.train_file[:, 3]
            self.lidar_RoI_points_path_list = [i.replace('image_02', cfg.DATA.RoI_POINTS_DIR).replace('png', 'pkl')
                                               for i in self.img_list]

        self.train_len = len(self.train_file)
        self.train_hw = cfg.TRAIN.IMAGE_HW
        self.sample_roi_points = cfg.TRAIN.SAMPLE_ROI_POINTS
        self.train_weight_file = cfg.TRAIN.WEIGHT_FILE

        self.batch_size = cfg.TRAIN.BATCH_SIZE

        self.dim_prior = cfg.DATA.DIM_PRIOR
        self.random_flip = cfg.TRAIN.FLIP

        self.__build_weights__(self.train_weight_file)

    def __build_weights__(self, weight_path):
        if os.path.exists(weight_path):
            return
        weights = [len(pickle.load(open(i, 'rb'))['bbox2d']) for i in self.lidar_RoI_points_path_list]
        np.savetxt(weight_path, np.array(weights))
        print('generate weights, done!, the valid training sample (the number of valid objects > 0) is : {}'.format(np.sum(np.array(weights) > 0)))


    def __getitem__(self, index):
        cv.setNumThreads(0)

        l_img_name = self.img_list[index]
        # load calib files
        if self.data_mode == 'KITTI 3D':
            calib = calib_parse.parse_calib('3d', self.calib_list[index])
        else:
            calib = calib_parse.parse_calib('raw', [self.calib_cam_to_cam_list[index], self.calib_velo_to_cam_list[index]])
        P2 = calib['P2']

        with open(self.lidar_RoI_points_path_list[index], 'rb') as f:
            RoI_box_points = pickle.load(f)

        # only use car by default
        cls_info = np.ones_like(RoI_box_points['bbox2d'][:, 0]) * 2
        bbox2d = RoI_box_points['bbox2d']
        RoI_points = RoI_box_points['RoI_points']

        l_img, bbox_shift = self.cut_or_pad_img(l_img_name, self.train_hw)
        bbox2d += bbox_shift

        '''random choose objects'''
        random_ind = np.random.randint(0, len(bbox2d), size=3)
        bbox2d = bbox2d[random_ind]
        cls_info = cls_info[random_ind].astype(np.int32)
        RoI_points = RoI_points[random_ind]

        batch_RoI_points = np.zeros((bbox2d.shape[0], self.sample_roi_points, 3), dtype=np.float32)
        batch_lidar_y_center = np.zeros((bbox2d.shape[0], 1), dtype=np.float32)
        batch_lidar_orient = np.zeros((bbox2d.shape[0], 1), dtype=np.float32)
        batch_lidar_density = np.zeros((bbox2d.shape[0], self.sample_roi_points), dtype=np.float32)
        batch_dim = []
        for i in range(bbox2d.shape[0]):
            y_coor = RoI_points[i][:, 1]
            batch_lidar_y_center[i] = np.mean(y_coor)
            y_thesh = (np.max(y_coor) + np.min(y_coor)) / 2
            y_ind = RoI_points[i][:, 1] > y_thesh

            y_ind_points = RoI_points[i][y_ind]
            if y_ind_points.shape[0] < 10:
                y_ind_points = RoI_points[i]

            rand_ind = np.random.randint(0, y_ind_points.shape[0], 100)
            depth_points_sample = y_ind_points[rand_ind]
            batch_RoI_points[i] = depth_points_sample
            depth_points_np_xz = depth_points_sample[:, [0, 2]]

            '''orient'''
            orient_set = [(i[1] - j[1]) / (i[0] - j[0]) for j in depth_points_np_xz
                          for i in depth_points_np_xz]
            orient_sort = np.array(sorted(np.array(orient_set).reshape(-1)))
            orient_sort = np.arctan(orient_sort[~np.isnan(orient_sort)])
            orient_sort_round = np.around(orient_sort, decimals=1)
            set_orenit = list(set(orient_sort_round))

            ind = np.argmax([np.sum(orient_sort_round == i) for i in set_orenit])
            orient = set_orenit[ind]
            if orient < 0:
                orient += np.pi

            if orient > np.pi / 2 + np.pi * 3 / 8:
                orient -= np.pi / 2
            if orient < np.pi / 8:
                orient += np.pi / 2

            if np.max(RoI_points[i][:, 0]) - np.min(RoI_points[i][:, 0]) > 4 and \
                    (orient >= np.pi / 8 and orient <= np.pi / 2 + np.pi * 3 / 8):
                if orient < np.pi / 2:
                    orient += np.pi / 2
                else:
                    orient -= np.pi / 2
            batch_lidar_orient[i] = orient

            '''density'''
            p_dis = np.array([(i[0] - depth_points_sample[:, 0]) ** 2 + (i[2] - depth_points_sample[:, 2]) ** 2
                                 for i in depth_points_sample])
            batch_lidar_density[i] = np.sum(p_dis < 0.04, axis=1)


            '''dim'''
            cls_dim_prior = self.dim_prior[cls_info[i]]
            batch_dim.append(cls_dim_prior)
        batch_dim = np.array(batch_dim)

        return {'P2': P2.astype(np.float32),
                'file_name': self.lidar_RoI_points_path_list[index],
                'l_img': np.transpose(l_img, [2, 0, 1]).astype(np.float32),
                'bbox2d': bbox2d.astype(np.float32),
                'cls_info': cls_info.astype(np.int32),
                'batch_RoI_points': batch_RoI_points.astype(np.float32),
                'batch_lidar_y_center': batch_lidar_y_center.astype(np.float32),
                'batch_lidar_orient': batch_lidar_orient.astype(np.float32),
                'batch_lidar_density': batch_lidar_density.astype(np.float32),
                'batch_dim': batch_dim.astype(np.float32),
                }

    def __len__(self):
        return self.train_len - self.train_len % self.batch_size


class KITTI3D_Object_Dataset_BBox2D(KITTI3D_Object_Dataset):
    def __init__(self, cfg):
        super(KITTI3D_Object_Dataset_BBox2D, self).__init__(cfg)
        self.det_2D_path = cfg.INFER.DET_2D_PATH
        self.root_3d_path = cfg.DATA.ROOT_3D_PATH

        self.infer_file = [i[:-4] for i in sorted(os.listdir(self.det_2D_path))]
        self.infer_len = len(self.infer_file)

        self.cls_list = cfg.DATA.CLS_LIST
        self.type_2_int = {i:j for i, j in zip(cfg.DATA.TYPE, range(len(cfg.DATA.TYPE)))}
        self.train_hw = (370, 1232)


    def __getitem__(self, index):
        cv.setNumThreads(0)
        file_name = self.infer_file[index]

        l_img_name = os.path.join(self.root_3d_path, 'image_2', file_name+'.png')
        l_img, bbox_shift = self.cut_or_pad_img(l_img_name, self.train_hw)


        # load calib files
        with open(os.path.join(self.root_3d_path, 'calib', file_name+'.txt'), encoding='utf-8') as f:
            text = f.readlines()
            P2 = np.array(text[2].split(' ')[1:], dtype=np.float32).reshape(3, 4)

        # load 2D detection files
        det_2D = np.loadtxt(os.path.join(self.det_2D_path, file_name+'.txt'), dtype=str).reshape(-1, 6)
        det_2D_ind = np.array([i in self.cls_list for i in det_2D[:, 0]])
        if len(det_2D_ind) < 1:
            return {'P2': np.array([]), 'file_name': file_name, 'l_img': np.array([]), 'det_2D': np.array([]), 'bbox2d': np.array([])}
        det_2D = det_2D[det_2D_ind]
        bbox2d = (det_2D[:, 1:5]).copy()

        bbox2d = bbox2d.astype(np.float32) + bbox_shift
        det_2D[:, 0] = np.array([self.type_2_int[i] for i in det_2D[:, 0]])

        return {'P2': P2.astype(np.float32),
                'file_name': file_name,
                'l_img': np.transpose(l_img, [2, 0, 1]).astype(np.float32),
                'det_2D': det_2D.astype(np.float32),
                'bbox2d': bbox2d.astype(np.float32)
                }

    def __len__(self):
        return self.infer_len