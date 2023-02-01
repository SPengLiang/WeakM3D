#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from yacs.config import CfgNode as CN

_C = CN()
_C.TRAIN = CN()
_C.VAL = CN()
_C.INFER = CN()
_C.DATA = CN()

_C.EXP_NAME = "default"
_C.NET_LAYER = 34
_C.RESTORE_PATH = None
_C.RESTORE_EPOCH = None

_C.LOG_DIR = './log'
_C.CHECKPOINTS_DIR = '/private/pengliang/WeakM3D_official/checkpoints'


_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.EPOCH = 50
_C.TRAIN.LR = 1e-4
_C.TRAIN.WEIGHT_FILE = '/private/pengliang/WeakM3D_official/data/kitti/data_file/kitti_raw_training_weight.txt'
_C.TRAIN.TRAIN_FILE = '/private/pengliang/WeakM3D_official/data/kitti/data_file/split/train_raw.txt'
_C.TRAIN.IMAGE_HW = (370, 1232)
_C.TRAIN.SAMPLE_ROI_POINTS = 5000
_C.TRAIN.SAMPLE_LOSS_POINTS = 100
_C.TRAIN.WORKS = 4
_C.TRAIN.FLIP = 0.0

_C.VAL.WORKS = 16
_C.VAL.SPLIT_FILE = '/private/pengliang/WeakM3D_official/data/kitti/data_file/split/val.txt'
_C.VAL.GT_DIR = '/private/pengliang/WeakM3D_official/data/kitti/KITTI3D/training/label_2'

_C.INFER.WORKS = 16
_C.INFER.DET_2D_PATH = '/private/pengliang/KITTI3D/training/rgb_detections/val/'
_C.INFER.SAVE_DIR = '/private/pengliang/WeakM3D_official/pred'


_C.DATA.CLS_LIST = ['Car']
_C.DATA.MODE = 'KITTI Raw'
_C.DATA.ROOT_3D_PATH = '/private/pengliang/WeakM3D_official/data/kitti/KITTI3D/training'
_C.DATA.RoI_POINTS_DIR = 'lidar_RoI_points'
_C.DATA.KITTI_RAW_PATH = '/private/pengliang/WeakM3D_official/data/kitti/raw_data'


_C.DATA.TYPE = ['Car', 'Cyclist', 'Pesdstrain']
_C.DATA.IMAGENET_STATS_MEAN = [0.485, 0.456, 0.406]
_C.DATA.IMAGENET_STATS_STD = [0.229, 0.224, 0.225]

_C.DATA.DIM_PRIOR = [[0.8, 1.8, 0.8], [0.6, 1.8, 1.8], [1.6, 1.8, 4.]]


