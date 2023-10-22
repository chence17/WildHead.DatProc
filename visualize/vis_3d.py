import cv2
import numpy as np

from face3d import mesh
from face3d.mesh_io.mesh import load_obj_mesh


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


def render_camera_check(cam, img_data, isBGR):
    if isBGR:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    r_img = render_camera(np.array(cam))
    h, w = r_img.shape[:2]
    a_img = cv2.resize(img_data, (h, w))
    vis_img = np.hstack([r_img, a_img])
    return vis_img
