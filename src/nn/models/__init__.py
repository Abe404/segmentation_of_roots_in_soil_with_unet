# pylint: disable=C0111, W0221, R0902
"""
U-Net architecture based on:
https://arxiv.org/abs/1505.04597
And modified to use Group Normalization
https://arxiv.org/abs/1803.08494


Copyright (C) 2019 Abraham Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch.nn as nn
import torch
import models.mask2former
import models.root_painter_unet
import models.sam
import models.segmentation_pytorch
import models.torchvision_models
import models.unet

import torch.nn.functional as F


model_map = {
    model_name: module
    for module in (
        torchvision_models, segmentation_pytorch, mask2former, root_painter_unet, sam, unet
    )
    for model_name in module.models
}


def get_model(name, encoder_name, pretrained_model=False, pretrained_backbone=False):
    return model_map[name].new(name, encoder_name, pretrained_model, pretrained_backbone)
