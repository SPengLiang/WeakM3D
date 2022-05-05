import os
import shutil
import numpy as np
from tqdm import tqdm
import torchvision
import cv2 as cv
import torch


kitti_raw_file_name = './data/kitti/data_file/split/train_raw.txt'
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


train_file = np.loadtxt(kitti_raw_file_name, dtype=str)
img_list = train_file[:, 0]

# pre_trained_mask_rcnn_path = '/private/pengliang/maskrcnn_resnet50_fpn.pth'
# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
# model.load_state_dict(torch.load(pre_trained_mask_rcnn_path))

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.cuda()
model.eval()


with torch.no_grad():
    for ind, _ in tqdm(enumerate(img_list)):
        seg_mask_path = img_list[ind].replace('image_02', 'seg_mask').replace('png', 'npz')
        seg_bbox_path = img_list[ind].replace('image_02', 'seg_bbox').replace('png', 'txt')

        seg_mask_dir = os.path.dirname(seg_mask_path)
        seg_bbox_dir = os.path.dirname(seg_bbox_path)
        if not os.path.exists(seg_mask_dir):
            os.makedirs(seg_mask_dir)
        if not os.path.exists(seg_bbox_dir):
            os.makedirs(seg_bbox_dir)

        image = (cv.imread(img_list[ind]))
        image_tensor = torchvision.transforms.functional.to_tensor(image)

        output = model([image_tensor.cuda()])[0]
        labels = output['labels'].cpu().numpy()

        '''for car'''
        ind = (labels == 3)
        scores = (output['scores'].cpu().numpy())
        score_ind = scores > 0.5
        ind = ind & score_ind

        bbox = (output['boxes'].cpu().numpy())[ind]
        masks = (output['masks'].cpu().numpy())[ind]
        scores = scores[ind]

        if bbox.shape[0] > 0:
            bbox2d = np.hstack([bbox.reshape(-1, 4), scores.reshape(-1, 1)])
            np.savetxt(seg_bbox_path, bbox2d, fmt='%f')
            np.savez(seg_mask_path, masks=masks[:, 0, ...])



