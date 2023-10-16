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
from scipy.spatial.transform import Rotation
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


def box2quad(box):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    quad = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]).astype(np.float32)
    return quad


def transform_box(box, scale, shift):
    x1, y1, w, h = box
    cx, cy = x1 + w / 2, y1 + h / 2
    w_ = h_ = max(w * scale[0], h * scale[1])
    cx_ = cx + shift[0] * w
    cy_ = cy + shift[1] * h
    x1_ = cx_ - w_ / 2
    y1_ = cy_ - h_ / 2
    return np.array([x1_, y1_, w_, h_]).astype(np.float32)


def R2hpose(R_matrix):
    """degree"""
    r_hpose = Rotation.from_matrix(R_matrix).as_euler('zyx', degrees=True)
    r_roll, r_yaw, r_pitch = r_hpose.astype(np.float32).tolist()
    pitch, yaw, roll = -r_pitch, -r_yaw, -r_roll
    hpose = [yaw, roll, pitch]
    return hpose

def hpose2R(hpose):
    yaw, roll, pitch = hpose
    r_pitch, r_yaw, r_roll = -pitch, -yaw, -roll
    R = Rotation.from_euler('zyx', [r_roll, r_yaw, r_pitch], degrees=True).as_matrix()
    return R

def hpose2camera(hpose):
    R = hpose2R(hpose)
    s = 1
    t3d = np.array([0., 0., 0.])
    R[:, :3] = R[:, :3] * s
    P = np.concatenate([R, t3d[:, None]], 1)
    P = np.concatenate([P, np.array([[0, 0, 0, 1.]])], 0)
    cam = eg3dcamparams(P.flatten())
    return cam


def calculate_y_angle(R):
    y_vector = R[:2, 1]
    y_normal = np.array([0., 1.])
    cos_ = np.dot(y_vector, y_normal)
    sin_ = np.cross(y_vector, y_normal)
    arctan2_ = np.arctan2(sin_, cos_)
    angle = np.degrees(arctan2_)
    return angle
