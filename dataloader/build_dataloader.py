import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from . import kitti_dataloader


def process_batch_data(sample):
    batch_data = {}
    keys = sample.keys()

    for k in keys:
        if k == 'file_name':
            batch_data[k] = sample[k]
        else:
            batch_data[k] = sample[k].cuda()

    return batch_data

def build_train_loader(cfg):
    train_dataset_RoI = kitti_dataloader.KITTI3D_Object_Dataset_Raw_RoI_Lidar(cfg)
    sampler_weights_RoI = np.loadtxt(cfg.TRAIN.WEIGHT_FILE)

    sampler_RoI = WeightedRandomSampler(weights=sampler_weights_RoI,
                                        num_samples=len(sampler_weights_RoI),
                                        replacement=True)
    TrainImgLoader_RoI = DataLoader(train_dataset_RoI,
                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                    num_workers=cfg.TRAIN.WORKS,
                                    sampler=sampler_RoI,
                                    pin_memory=True,
                                    drop_last=True)
    return TrainImgLoader_RoI


def build_infer_loader(cfg):
    infer_dataset = kitti_dataloader.KITTI3D_Object_Dataset_BBox2D(cfg)
    InferImgLoader = DataLoader(infer_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=cfg.INFER.WORKS,
                                drop_last=False)
    return InferImgLoader


