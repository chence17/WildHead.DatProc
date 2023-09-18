import cv2
import json
import math
import torch
import argparse
import onnxruntime
import numpy as np
import os.path as osp
from scipy.spatial.transform import Rotation


from utils.face_parsing import show_image, HeadParser
from recrop_images import crop_final, eg3dcamparams
from face3d import mesh
from face3d.mesh_io.mesh import load_obj_mesh
from utils.dataset_process import find_meta_files

def parse_args():
    parser = argparse.ArgumentParser(description='Run Frontal View Pipeline')
    parser.add_argument('json_dir', type=str, help='path to json file directory',
                        default='/home/shitianhao/project/DatProc/utils/stats/')
    args, _ = parser.parse_known_args()
    return args


def render_image(R, vertices, triangles, colors):
    vertices_ = vertices - np.mean(vertices, 0)[np.newaxis, :]
    s = 1.25 * 180 / (np.max(vertices_[:, 1]) - np.min(vertices_[:, 1]))
    t = [0, 0, 0]
    vertices_ = mesh.transform.similarity_transform(vertices_, s, R, t)
    h = w = 256
    bg_img = np.ones([h, w, 3], dtype=np.float32) * 0.5
    image_vertices = mesh.transform.to_image(vertices_, h, w)
    rendering_cc = mesh.render.render_colors(
        image_vertices, triangles, colors, h, w, BG=bg_img)
    rendering_cc_u8 = (rendering_cc * 255).astype(np.uint8)
    return rendering_cc_u8


def normalize(v):
    return v / np.linalg.norm(v)


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


def rotate_point(point, center, angle):
    """
    计算点绕固定点旋转一定角度后的新坐标
    :param point: 要旋转的点的坐标 (x, y)
    :param center: 固定点的坐标 (x, y)
    :param angle: 旋转角度（弧度制）
    :return: 旋转后的新坐标 (x', y')
    """
    x, y = point
    cx, cy = center

    # 将角度转换为弧度
    angle_rad = math.radians(angle)

    # 计算旋转后的新坐标
    new_x = (x - cx) * math.cos(angle_rad) - \
        (y - cy) * math.sin(angle_rad) + cx
    new_y = (x - cx) * math.sin(angle_rad) + \
        (y - cy) * math.cos(angle_rad) + cy

    return new_x, new_y


class WHENetHeadPoseEstimator(object):
    def __init__(self, weights_file: str, input_width: int = 224, input_height: int = 224) -> None:
        self.weights_file = weights_file
        self.providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            self.providers.insert(0, 'CUDAExecutionProvider')
        self.estimator = onnxruntime.InferenceSession(
            self.weights_file, providers=self.providers)
        self.input_width = input_width
        self.input_height = input_height
        self.input_hw = (self.input_height, self.input_width)
        self.input_name = self.estimator.get_inputs()[0].name
        self.output_names = [
            output.name for output in self.estimator.get_outputs()]
        self.output_shapes = [
            output.shape for output in self.estimator.get_outputs()]

    def __call__(self, image_data: np.ndarray, isBGR: bool) -> np.ndarray:
        if isBGR:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        chw = image_data.transpose(2, 0, 1)
        nchw = np.asarray(chw[np.newaxis, :, :, :], dtype=np.float32)
        outputs = self.estimator.run(
            output_names=self.output_names,
            input_feed={self.input_name: nchw}
        )
        yaw, roll, pitch = outputs[0][0][0], outputs[0][0][1], outputs[0][0][2]
        yaw, roll, pitch = np.squeeze([yaw, roll, pitch])
        return np.array([yaw, roll, pitch])


def show_hbox(img_data, hbox, is_bgr, title, show_axis=True):
    x1, y1, w, h = hbox
    x2, y2 = x1 + w, y1 + h
    hbox_img = cv2.rectangle(img_data, (x1, y1), (x2, y2), (0, 0, 255), 2)
    show_image(hbox_img, is_bgr, title, show_axis)


def calculate_R(hpose):
    yaw, roll, pitch = hpose
    r_pitch, r_yaw, r_roll = -pitch, -yaw, -roll
    R = Rotation.from_euler(
        'zyx', [r_roll, r_yaw, r_pitch], degrees=True).as_matrix()
    return R


def calculate_y_angle(R):
    y_vector = R[:2, 1]
    y_normal = np.array([0., 1.])
    cos_ = np.dot(y_vector, y_normal)
    sin_ = np.cross(y_vector, y_normal)
    arctan2_ = np.arctan2(sin_, cos_)
    angle = np.degrees(arctan2_)
    return angle


def rotate_image(image, center, angle, top_expand=0.1, left_expand=0.05,
                 bottom_expand=0.0, right_expand=0.05, blur_kernel=None,
                 border_mode=cv2.BORDER_REFLECT):
    height, width = image.shape[:2]
    crop_w = int(width * (1 + left_expand + right_expand))
    crop_h = int(height * (1 + top_expand + bottom_expand))
    crop_size = (crop_w, crop_h)
    size = max(height, width)
    top = int(height * top_expand)
    left = int(width * left_expand)
    bound = np.array([[left, top], [left, top + height - 1],
                      [left + width - 1, top + height - 1], [left + width - 1, top]],
                     dtype=np.float32)
    quad = np.array([[0, 0], [0, crop_h - 1], [crop_w - 1, crop_h - 1], [crop_w - 1, 0]],
                    dtype=np.float32)
    quad = rotate_quad(quad, angle, center)

    # Generate rotation matrix
    mat = cv2.getAffineTransform(quad[:3], bound[:3])

    # Apply rotation to the image
    rotated_image = cv2.warpAffine(
        image, mat, crop_size, flags=cv2.INTER_LANCZOS4, borderMode=border_mode)
    empty = np.ones_like(image) * 255
    crop_mask = cv2.warpAffine(empty, mat, crop_size)
    mask_kernel = int(size * 0.02) * 2 + 1
    blur_kernel = int(size * 0.03) * 2 + \
        1 if blur_kernel is None else blur_kernel
    if crop_mask.mean() < 255:
        blur_mask = cv2.blur(crop_mask.astype(np.float32).mean(
            2), (mask_kernel, mask_kernel)) / 255.0
        blur_mask = blur_mask[..., np.newaxis]  # .astype(np.float32) / 255.0
        blurred_img = cv2.blur(rotated_image, (blur_kernel, blur_kernel), 0)
        rotated_image = rotated_image * \
            blur_mask + blurred_img * (1 - blur_mask)
        rotated_image = rotated_image.astype(np.uint8)
    return rotated_image


def get_hbox_image(hbox, img_data):
    x1, y1, w, h = hbox
    x2, y2 = x1 + w, y1 + h
    hbox_img_center = (w / 2, h / 2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_data.shape[1], x2), min(img_data.shape[0], y2)
    hbox_img = img_data[y1:y2, x1:x2, :]
    return hbox_img, hbox_img_center


def rotate_quad(hbox_quad, angle, center):
    hbox_quad_ = []
    for i in hbox_quad:
        i_rot = rotate_point(i, center, angle)
        hbox_quad_.append(i_rot)
    return np.array(hbox_quad_).astype(np.float32)


def get_rotate_angle(hbox, img_data, is_bgr, pe, iterations=3):
    x1, y1, w, h = hbox
    x2, y2 = x1 + w, y1 + h
    hbox_quad = np.array([[x1, y1], [x1, y2], [x2, y2],
                         [x2, y1]]).astype(np.float32)
    hbox_center = np.mean(hbox_quad, axis=0)
    angle = 0
    for _ in range(iterations):
        hbox_quad_rot = rotate_quad(hbox_quad.copy(), angle, hbox_center)
        hbox_img = crop_final(img_data, size=pe.input_height, quad=hbox_quad_rot,
                              top_expand=0., left_expand=0., bottom_expand=0., right_expand=0.)
        hpose = pe(hbox_img, isBGR=is_bgr)
        R = calculate_R(hpose)
        angle += calculate_y_angle(R)
    return angle, hbox_center


def get_scaled_hbox(ratio, msk_hbox, hbox):
    assert ratio > 1.0, 'ratio must be greater than 1.0'
    h, w = msk_hbox.shape[:2]
    msk_hbox = cv2.threshold(msk_hbox, 127, 255, cv2.THRESH_BINARY)[1]
    output = cv2.connectedComponentsWithStats(msk_hbox, connectivity=4)[1]
    unique, counts = np.unique(output, return_counts=True)
    unique, counts = unique[1:], counts[1:]  # discard background
    max_idx = np.argmax(counts)
    result = np.where(output == unique[max_idx], 255, 0).astype(np.uint8)
    head_top = np.argmax(np.any(result, axis=1))
    h_ = int((h - head_top) * ratio)
    x1, y1, hbox_w, hbox_h = hbox
    hbox_cx, hbox_cy = x1 + hbox_w / 2, y1 + hbox_h / 2
    x1_, y1_ = int(hbox_cx - h_ / 2), int(hbox_cy - h_ / 2)
    hbox_ = [x1_, y1_, h_, h_]
    return hbox_


def hbox2quad(hbox):
    x1, y1, w, h = hbox
    x2, y2 = x1 + w, y1 + h
    quad = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
                    ).astype(np.float32)
    return quad


data_root = 'samples'
json_path = 'samples/meta_1-1.json'
vertices, triangles, colors = load_obj_mesh('assets/yaoming.obj')

fp = HeadParser()

pe = WHENetHeadPoseEstimator('assets/whenet_1x3x224x224_prepost.onnx')
assert pe.input_height == pe.input_width

ratio = 1.05
crop_size = 563


def main(args):

    with open(json_path, 'r') as f:
        meta = json.load(f)

    for img_name in meta.keys():
        if img_name not in ['images/000031.png', 'images/000679.png', 'images/000320.png']:
            continue
        img_path = osp.join(data_root, img_name)
        img_data = cv2.imread(img_path)
        for k, v in meta[img_name].items():
            hbox = v['head_box']

            show_hbox(img_data.copy(), hbox, True,
                      f'hbox {img_name} {k}', True)

            # cx, cy = x1 + w / 2, y1 + h / 2
            angle, _ = get_rotate_angle(
                hbox, img_data.copy(), True, pe, iterations=3)
            img_hbox, _ = get_hbox_image(hbox, img_data.copy())
            # img_hbox_rot = rotate_image(img_hbox.copy(), hbox_img_center, angle)
            show_image(img_hbox, True,
                       f'img_hbox {img_name} {k}', show_axis=True)
            # show_image(img_hbox_rot, True, f'img_hbox_rot {img_name} {k}', show_axis=True)
            sem_hbox = fp(img_hbox, True, False)
            msk_hbox = (sem_hbox != 0).astype(np.uint8) * 255
            hbox = get_scaled_hbox(ratio, msk_hbox, hbox)
            hbox_quad = hbox2quad(hbox)
            hbox_center = np.mean(hbox_quad, axis=0)
            # save
            hbox_quad = rotate_quad(hbox_quad.copy(), angle, hbox_center)
            # save
            hbox_img = crop_final(img_data.copy(), size=crop_size, quad=hbox_quad,
                                  top_expand=0., left_expand=0., bottom_expand=0., right_expand=0.)
            show_image(hbox_img, True,
                       f'hbox_img {img_name} {k}', show_axis=True)
            print(hbox_img.shape)
            # save
            hbox_img_sem = fp(hbox_img, True, False)
            hbox_img_hpose = pe(cv2.resize(hbox_img, pe.input_hw), isBGR=True)
            R = calculate_R(hbox_img_hpose)
            R_img = render_image(R, vertices.copy(),
                                 triangles.copy(), colors.copy())
            show_image(
                R_img, False, f'R_raw_img {img_name} {k}', show_axis=True)
            s = 1
            t3d = np.array([0., 0., 0.])
            R[:, :3] = R[:, :3] * s
            P = np.concatenate([R, t3d[:, None]], 1)
            P = np.concatenate([P, np.array([[0, 0, 0, 1.]])], 0)
            # save
            cam = eg3dcamparams(P.flatten())
            print(cam)
