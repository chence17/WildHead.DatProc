'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-09-13 19:30:32
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-09-13 19:49:16
FilePath: /DatProc/TDDFA_V2/FaceBoxes/utils/nms_wrapper.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# coding: utf-8

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from TDDFA_V2.FaceBoxes.utils.nms.cpu_nms import cpu_nms, cpu_soft_nms


def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    return cpu_nms(dets, thresh)
    # return gpu_nms(dets, thresh)
