'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-09-18 23:20:23
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-10-18 19:40:15
FilePath: /DatProc/utils/tool.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import math
import numpy as np

from face3d import mesh
from face3d.mesh_io.mesh import load_obj_mesh
# from utils.recrop_images import eg3dcamparams


def partition_dict(d, partition_size):
    for i in range(0, len(d), partition_size):
        yield dict(list(d.items())[i:i + partition_size])


