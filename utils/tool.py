'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-09-18 23:20:23
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-09-19 22:57:40
FilePath: /DatProc/utils/tool.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import math
import numpy as np
from scipy.spatial.transform import Rotation
from face3d import mesh
from face3d.mesh_io.mesh import load_obj_mesh

def partition_dict(d, partition_size):
    for i in range(0, len(d), partition_size):
        yield dict(list(d.items())[i:i + partition_size])


def draw_detection_box(image_data, box, text):
    # Determine the detection box coordinates
    x, y, width, height = box
    scale = math.ceil(math.sqrt(image_data.shape[0] * image_data.shape[1]) / 512)
    thinkness = scale * 2
    font_scale = scale * 0.9
    offset_scale = scale * 10

    # Draw the detection box
    cv2.rectangle(image_data, (x, y), (x + width, y + height), (0, 255, 0), thinkness)

    # Add a text label
    cv2.putText(image_data, text, (x, y - offset_scale), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 255, 0), thinkness)
    return image_data


def draw_facial_landmarks(image_data, landmarks, color=(0, 255, 0)):
    scale = math.ceil(math.sqrt(image_data.shape[0] * image_data.shape[1]) / 512)
    radius = scale * 2
    for landmark in landmarks:
        x, y = landmark
        cv2.circle(image_data, (x, y), radius, color, -1)
    return image_data


def draw_quad(image_data, quad, color=(0, 255, 0)):
    scale = math.ceil(math.sqrt(image_data.shape[0] * image_data.shape[1]) / 512)
    thickness = scale * 2
    cv2.polylines(image_data, [quad], isClosed=True, color=color, thickness=thickness)
    return image_data


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


def hbox2quad(hbox):
    x1, y1, w, h = hbox
    x2, y2 = x1 + w, y1 + h
    quad = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]).astype(np.float32)
    return quad


def R2hpose(R_matrix):
    """degree"""
    r_hpose = Rotation.from_matrix(R_matrix).as_euler('zyx', degrees=True)
    r_roll, r_yaw, r_pitch = r_hpose.astype(np.float32).tolist()
    pitch, yaw, roll = -r_pitch, -r_yaw, -r_roll
    hpose = [yaw, roll, pitch]
    return hpose
