'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-09-18 23:20:23
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-10-16 10:38:37
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


def render_image(R, vertices, triangles, colors):
    vertices_ = vertices - np.mean(vertices, 0)[np.newaxis, :]
    s = 1.25 * 180 / (np.max(vertices_[:, 1]) - np.min(vertices_[:, 1]))
    t = [0, 0, 0]
    vertices_ = mesh.transform.similarity_transform(vertices_, s, R, t)
    h = w = 256
    bg_img = np.ones([h, w, 3], dtype=np.float32) * 0.5
    image_vertices = mesh.transform.to_image(vertices_, h, w)
    rendering_cc = mesh.render.render_colors(image_vertices, triangles, colors, h, w, BG=bg_img)
    rendering_cc_u8 = (rendering_cc * 255).astype(np.uint8)
    return rendering_cc_u8


def render_camera(panohead_camera):
    vertices, triangles, colors = load_obj_mesh('assets/yaoming.obj')
    c2wR = panohead_camera[:16].reshape(4, 4)[:3, :3]
    # inverse convert from OpenCV camera
    convert = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]).astype(np.float32)
    inv_convert = np.linalg.inv(convert)
    w2cR = inv_convert @ c2wR
    R_img = render_image(w2cR, vertices, triangles, colors)
    return R_img
