import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import torchvision
import torchvision.models as models

class ResnetEncoder(nn.Module):
    def __init__(self, num_layers=18, pretrained=True):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        res_feat_chs = {18: 256,
                        34: 256,
                        50: 1024}

        self.res_feat_chs = res_feat_chs[num_layers]

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.location_xy = nn.Sequential(
                    nn.Linear(self.res_feat_chs * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Linear(256, 2),
                )
        self.location_z = nn.Sequential(
            nn.Linear(self.res_feat_chs * 7 * 7, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
        )
        self.orientation_conf = nn.Sequential(
            nn.Linear(self.res_feat_chs * 7 * 7, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 2),
        )


    def forward(self, input_image, bbox):
        self.features = []
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))

        last_feat = self.features[-1]
        if len(bbox.shape) == 3:
            f = torchvision.ops.roi_align(last_feat, [i/16 for i in bbox], (7, 7))
        else:
            f = torchvision.ops.roi_align(last_feat, [bbox/16], (7, 7))

        f = f.view(-1, self.res_feat_chs * 7 * 7)

        location_xy = self.location_xy(f)
        location_xy = location_xy.view(-1, 2)

        location_z = self.location_z(f)
        orientation_conf = self.orientation_conf(f)

        return location_xy, location_z, orientation_conf