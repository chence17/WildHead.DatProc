'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-03-27 14:15:48
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-03-27 14:26:04
FilePath: /DataProcess/ibug/face_parsing/resnet/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
Backbone modules.
"""
import logging

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from ibug.face_parsing.resnet.decoder import *

_logger = logging.getLogger(__name__)


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module,  num_channels: int):
        super().__init__()

        return_layers = {"layer1": "c1", "layer2": "c2",
                         "layer3": "c3", "layer4": "c4"}
        self.body = IntermediateLayerGetter(
            backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, images, rois=None):
        return self.body(images)


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str):
        if 'resnet18' in name or 'resnet34' in name:
            replace_stride_with_dilation = [False, False, False]
        else:
            replace_stride_with_dilation = [False, True, True]
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=replace_stride_with_dilation)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, num_channels)
