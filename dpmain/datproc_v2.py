'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-10-16 13:48:12
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2024-02-28 23:49:01
FilePath: /DatProc/dpmain/datproc_v1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import cv2
import os.path as osp
import json

from scipy.spatial.transform import Rotation
from skimage import io

from dpfilter import ImageSizeFilter, ImageBlurFilter
from dpdetector import HeadDetector, FaceAlignmentDetector
from dpcropper import FrontViewCropperV2
# from dpparser import HeadParser, HeadSegmenter
from dpestimator import HeadPoseEstimator
import cv2
import numpy as np


def polygon2mask(polygon_str, img_shape):
    mask = np.zeros(img_shape, dtype=np.uint8)
    try:
        points = []
        k = json.loads(polygon_str)
        if len(k) == 0:
            return mask

        if len(k) == 1:
            k=k[0]
        for i in k:
            points.append([i['x'], i['y']])

        if len(points) == 0:
            return mask

        return cv2.fillPoly(mask, np.int32([points]), (255,255,255))
    except Exception as e:
        print(f"polygon2mask error: {e}")
        return mask


def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


class DatProcV2(object):
    def __init__(self, data_source) -> None:
        self.img_sz_flt = ImageSizeFilter(size_thres=512)
        self.img_br_flt = ImageBlurFilter(svd_thres=0.6, lap_thres=100)
        self.hed_det = HeadDetector(weights_file='assets/224x224_yolov4_hddet_480x640.onnx', input_width=640,
                                    input_height=480, size_thres=512)
        self.flmk_det = FaceAlignmentDetector(score_thres=0.8)
        self.fv_crpr = FrontViewCropperV2(config_file='TDDFA_V2/configs/mb1_120x120.yml', mode='gpu')
        # self.hed_par = HeadParser()
        # self.hed_seg = HeadSegmenter(use_fsam=True)
        # self.hed_seg = HeadSegmenter(use_fsam=False)
        self.hed_pe = HeadPoseEstimator(weights_file='assets/whenet_1x3x224x224_prepost.onnx')

        # inverse convert from OpenCV camera
        self.convert = np.array([[1, 0, 0, 0],
                                 [0, -1, 0, 0],
                                 [0, 0, -1, 0],
                                 [0, 0, 0, 1]]).astype(np.float32)
        self.inv_convert = np.linalg.inv(self.convert)

        # mean transform for q2b
        self.mean_q2b_scale = [0.7417686609206039, 0.7417686609206039]
        self.mean_q2b_shift = [-0.007425799169690871, 0.00886478197975557]

        # head image size and crop size
        self.head_image_size = 1024
        self.crop_size = self.get_final_crop_size(512)

        # data source
        self.data_source = data_source

    def __call__(self, img_path, json_path, use_landmarks=False):
        # Check if image path exists
        assert osp.exists(img_path), f"Image file not found: {img_path}"
        assert osp.exists(json_path), f"JSON file not found: {json_path}"

        # # Check if image is too small
        if not self.img_sz_flt(img_path):
            raise ValueError('Image size too small')

        # Load image data
        img_data = io.imread(img_path)  # RGB uint8 HW3 ndarray
        json_data = load_json(json_path)

        try:
            polygon1_mask = polygon2mask(json_data['polygon1'], img_data.shape) # Hair
        except Exception as e:
            print(f"polygon1_mask error: {e}")
            polygon1_mask = np.zeros(img_data.shape, dtype=np.uint8)

        try:
            polygon2_mask = polygon2mask(json_data['polygon2'], img_data.shape) # Face
        except Exception as e:
            print(f"polygon2_mask error: {e}")
            polygon2_mask = np.zeros(img_data.shape, dtype=np.uint8)

        # Detect head box
        hed_boxes = self.hed_det(img_data, isBGR=False, max_box_num=1) # N * 4 np.ndarray, each box is [x_min, y_min, w, h].
        if hed_boxes is None or hed_boxes.shape[0] == 0:
            raise ValueError('No head detected')
        box_np = np.array(hed_boxes[0])  # Coords in Raw Image

        # Get head box and landmarks
        if use_landmarks:
            head_image, _ = self.crop_head_image(img_data.copy(), box_np)
            assert head_image.shape[0] == head_image.shape[1]
            landmarks = self.flmk_det(head_image, isBGR=False, image_upper_left=box_np[:2])  # Coords in Raw Image
        else:
            landmarks = None

        # Load mask data
        img_msk_data = polygon1_mask

        # Load parsing data
        img_par_data = polygon2_mask

        # Process image with landmarks (front) or without landmarks (back)
        if landmarks is None:
            info_dict, head_image, head_image_par, head_image_msk, head_pad_mask, cropped_img, cropped_img_par, cropped_img_msk, cropped_pad_mask = self.process_back_view(img_data, img_msk_data, img_par_data, box_np)
        else:
            info_dict, head_image, head_image_par, head_image_msk, head_pad_mask, cropped_img, cropped_img_par, cropped_img_msk, cropped_pad_mask = self.process_front_view(img_data, img_msk_data, img_par_data, box_np, landmarks)

        info_dict['data_source'] = self.data_source  # data source
        info_dict['raw']['image'] = osp.basename(img_path)  # raw image name
        info_dict['raw']['box'] = box_np.tolist()  # [x_min, y_min, w, h]

        cur_cam = info_dict['head']['camera']
        cur_TMatrix = np.array(cur_cam[:16]).reshape(4, 4)
        cur_cam_scoord = self.get_cam_coords(cur_TMatrix)
        info_dict['head']["camera_scoord"] = cur_cam_scoord  # [theta, phi, r, x, y, z]
        # front [45, 135]
        # right [-45, 45]
        # back [-135, -45]
        # left [-180, -135], [135, 180]
        theta = cur_cam_scoord[0]
        if theta >= -45 and theta <= 45:
            info_dict['head']['view'] = 'right'
        elif theta >= 45 and theta <= 135:
            info_dict['head']['view'] = 'front'
        elif theta >= -135 and theta <= -45:
            info_dict['head']['view'] = 'back'
        else:
            info_dict['head']['view'] = 'left'

        svd_score, lap_score = self.img_br_flt.get_blur_degree(cropped_img.copy(), isBGR=False)
        info_dict['head']['svd_score'] = svd_score  # SVD score for blur detection
        info_dict['head']['laplacian_score'] = lap_score  # Laplacian score for blur detection

        info_dict['head']['par_ratio'] = np.sum(cropped_img_par != 0)/cropped_img_par.size  # Valid Area / Image Size. Valid Area: area of non-void head parsing pixels in the image which is not background in the parsing.
        info_dict['head']['msk_ratio'] = np.sum(cropped_img_msk != 0)/cropped_img_msk.size  # Valid Area / Image Size. Valid Area: area of non-void head parsing pixels in the image which is not background in the parsing.

        return info_dict, head_image, head_image_par, head_image_msk, head_pad_mask, cropped_img, cropped_img_par, cropped_img_msk, cropped_pad_mask

    @staticmethod
    def get_final_crop_size(size, top_expand=0.1, left_expand=0.05, bottom_expand=0.0, right_expand=0.05):
        crop_w = int(size * (1 + left_expand + right_expand))
        crop_h = int(size * (1 + top_expand + bottom_expand))
        assert crop_w == crop_h
        return crop_w

    @staticmethod
    def transform_box(box, scale, shift):
        x1, y1, w, h = box
        cx, cy = x1 + w / 2, y1 + h / 2
        w_ = h_ = max(w * scale[0], h * scale[1])
        cx_ = cx + shift[0] * w
        cy_ = cy + shift[1] * h
        x1_ = cx_ - w_ / 2
        y1_ = cy_ - h_ / 2
        return np.array([x1_, y1_, w_, h_]).astype(np.float32)

    @staticmethod
    def box2quad(box):
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        quad = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]).astype(np.float32)
        return quad

    @staticmethod
    def crop_head_image(image_data, box, bg_value=0):
        # expected input: [x_min, y_min, w, h]
        # print('box', box)
        x_min, y_min, w, h = np.int32(box)
        x_max, y_max = x_min + w, y_min + h
        # print(x_min, y_min, x_max, y_max)
        top_size, bottom_size, left_size, right_size = 0, 0, 0, 0
        if x_min < 0:
            left_size = abs(x_min)
            x_min = 0
        if y_min < 0:
            top_size = abs(y_min)
            y_min = 0
        if x_max > image_data.shape[1]:
            right_size = abs(x_max - image_data.shape[1])
            x_max = image_data.shape[1]
        if y_max > image_data.shape[0]:
            bottom_size = abs(y_max - image_data.shape[0])
            y_max = image_data.shape[0]
        # print(top_size, bottom_size, left_size, right_size)
        crop_img = image_data[y_min:y_max, x_min:x_max]
        # print(crop_img.shape)
        crop_msk = np.zeros_like(crop_img)  # 0: Not padding, 255: Padding
        if top_size + bottom_size + left_size + right_size != 0:
            crop_img = cv2.copyMakeBorder(crop_img, top_size, bottom_size, left_size, right_size,
                                          borderType=cv2.BORDER_REFLECT, value=bg_value)
            crop_msk = cv2.copyMakeBorder(crop_msk, top_size, bottom_size, left_size, right_size,
                                          borderType=cv2.BORDER_CONSTANT, value=255)  # 0: Not padding, 255: Padding
        # print(crop_img.shape, (h, w))
        assert crop_img.shape[0] == h
        assert crop_img.shape[1] == w
        return crop_img, crop_msk

    @staticmethod
    def crop_head_parsing(image_data, box, bg_value=0):
        # expected input: [x_min, y_min, w, h]
        # print('box', box)
        x_min, y_min, w, h = np.int32(box)
        x_max, y_max = x_min + w, y_min + h
        # print(x_min, y_min, x_max, y_max)
        top_size, bottom_size, left_size, right_size = 0, 0, 0, 0
        if x_min < 0:
            left_size = abs(x_min)
            x_min = 0
        if y_min < 0:
            top_size = abs(y_min)
            y_min = 0
        if x_max > image_data.shape[1]:
            right_size = abs(x_max - image_data.shape[1])
            x_max = image_data.shape[1]
        if y_max > image_data.shape[0]:
            bottom_size = abs(y_max - image_data.shape[0])
            y_max = image_data.shape[0]
        # print(top_size, bottom_size, left_size, right_size)
        crop_img = image_data[y_min:y_max, x_min:x_max]
        # print(crop_img.shape)
        if top_size + bottom_size + left_size + right_size != 0:
            crop_img = cv2.copyMakeBorder(crop_img, top_size, bottom_size, left_size, right_size,
                                          borderType=cv2.BORDER_CONSTANT, value=bg_value)
        # print(crop_img.shape, (h, w))
        assert crop_img.shape[0] == h
        assert crop_img.shape[1] == w
        return crop_img

    @staticmethod
    def R2hpose(R_matrix):
        """degree"""
        r_hpose = Rotation.from_matrix(R_matrix).as_euler('zyx', degrees=True)
        r_roll, r_yaw, r_pitch = r_hpose.astype(np.float32).tolist()
        pitch, yaw, roll = -r_pitch, -r_yaw, -r_roll
        hpose = [yaw, roll, pitch]
        return hpose

    @staticmethod
    def hpose2R(hpose):
        yaw, roll, pitch = hpose
        r_pitch, r_yaw, r_roll = -pitch, -yaw, -roll
        R = Rotation.from_euler('zyx', [r_roll, r_yaw, r_pitch], degrees=True).as_matrix()
        return R

    @staticmethod
    def calculate_y_angle(R):
        y_vector = R[:2, 1]
        y_normal = np.array([0., 1.])
        cos_ = np.dot(y_vector, y_normal)
        sin_ = np.cross(y_vector, y_normal)
        arctan2_ = np.arctan2(sin_, cos_)
        angle = np.degrees(arctan2_)
        return angle

    @staticmethod
    def rotate_image(img, center, angle, borderMode=cv2.BORDER_REFLECT, upsample=2):
        rotmat = cv2.getRotationMatrix2D(center, angle, 1)
        crop_w = img.shape[1]
        crop_h = img.shape[0]
        crop_size = (crop_w, crop_h)
        if upsample is None or upsample == 1:
            crop_img = cv2.warpAffine(np.array(img), rotmat, crop_size, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
        else:
            assert isinstance(upsample, int)
            crop_size_large = (crop_w * upsample, crop_h * upsample)
            crop_img = cv2.warpAffine(np.array(img), upsample * rotmat, crop_size_large, flags=cv2.INTER_LANCZOS4,
                                      borderMode=borderMode)
            crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_AREA)

        empty = np.ones_like(img) * 255
        crop_mask = cv2.warpAffine(empty, rotmat, crop_size)

        size = min(crop_w, crop_h)
        mask_kernel = int(size*0.02)*2+1
        blur_kernel = int(size*0.03)*2+1

        if crop_mask.mean() < 255:
            blur_mask = cv2.blur(crop_mask.astype(np.float32).mean(2), (mask_kernel, mask_kernel)) / 255.0
            blur_mask = blur_mask[..., np.newaxis]  # .astype(np.float32) / 255.0
            blurred_img = cv2.blur(crop_img, (blur_kernel, blur_kernel), 0)
            crop_img = crop_img * blur_mask + blurred_img * (1 - blur_mask)
            crop_img = crop_img.astype(np.uint8)

        return crop_img, rotmat

    @staticmethod
    def rotate_parsing(img, center, angle, upsample=2, bg_value=0):
        rotmat = cv2.getRotationMatrix2D(center, angle, 1)
        crop_w = img.shape[1]
        crop_h = img.shape[0]
        crop_size = (crop_w, crop_h)
        if upsample is None or upsample == 1:
            crop_img = cv2.warpAffine(np.array(img), rotmat, crop_size, flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)
        else:
            assert isinstance(upsample, int)
            crop_size_large = (crop_w * upsample, crop_h * upsample)
            crop_img = cv2.warpAffine(np.array(img), upsample * rotmat, crop_size_large, flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=bg_value)
            crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_NEAREST)

        return crop_img, rotmat

    @staticmethod
    def rotate_image_and_quad(img, quad, tf_quad, borderMode=cv2.BORDER_REFLECT, upsample=2):
        mat = cv2.getAffineTransform(tf_quad[:3], quad[:3])
        angle = np.degrees(np.arctan2(mat[1, 0], mat[0, 0]))
        center = np.mean(quad[:3], axis=0)
        rotmat = cv2.getRotationMatrix2D(center, angle, 1)
        crop_w = img.shape[1]
        crop_h = img.shape[0]
        crop_size = (crop_w, crop_h)
        if upsample is None or upsample == 1:
            crop_img = cv2.warpAffine(np.array(img), rotmat, crop_size, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
        else:
            assert isinstance(upsample, int)
            crop_size_large = (crop_w * upsample, crop_h * upsample)
            crop_img = cv2.warpAffine(np.array(img), upsample * rotmat, crop_size_large, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
            crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_AREA)

        empty = np.ones_like(img) * 255
        crop_mask = cv2.warpAffine(empty, rotmat, crop_size)

        size = min(crop_w, crop_h)
        mask_kernel = int(size*0.02)*2+1
        blur_kernel = int(size*0.03)*2+1

        if crop_mask.mean() < 255:
            blur_mask = cv2.blur(crop_mask.astype(np.float32).mean(2), (mask_kernel, mask_kernel)) / 255.0
            blur_mask = blur_mask[..., np.newaxis]  # .astype(np.float32) / 255.0
            blurred_img = cv2.blur(crop_img, (blur_kernel, blur_kernel), 0)
            crop_img = crop_img * blur_mask + blurred_img * (1 - blur_mask)
            crop_img = crop_img.astype(np.uint8)

        rot_quad = cv2.transform(quad.reshape(1, -1, 2), rotmat).reshape(-1, 2)
        return crop_img, rotmat, rot_quad

    @staticmethod
    def rotate_quad(quad, center, angle):
        rotmat = cv2.getRotationMatrix2D(center, angle, 1)
        rot_quad = cv2.transform(quad.reshape(1, -1, 2), rotmat).reshape(-1, 2)
        return rot_quad

    @staticmethod
    def landmarks_to_bbox(landmarks, margin=0.75):
        # Find minimum and maximum x and y coordinates
        min_x = np.min(landmarks[:, 0])
        max_x = np.max(landmarks[:, 0])
        min_y = np.min(landmarks[:, 1])
        max_y = np.max(landmarks[:, 1])

        # Calculate width and height of bounding box
        width = max_x - min_x
        height = max_y - min_y

        # Add margin to bounding box
        x_margin = margin * width
        y_margin = margin * height
        min_x -= x_margin
        max_x += x_margin
        min_y -= 1.5 * y_margin
        max_y += 0.5 * y_margin

        # bounding box coordinates
        min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
        cx, cy = (min_x + max_x) / 2., (min_y + max_y) / 2.
        width = max(max_x - min_x, max_y - min_y)
        height = width
        min_x, min_y = int(cx - width / 2.), int(cy - height / 2.)

        # Return bounding box coordinates
        return np.array([min_x, min_y, width, height])

    @staticmethod
    def get_cam_coords(c2w):
        # World Coordinate System: x(right), y(up), z(forward)
        T = c2w[:3, 3]
        x, y, z = T
        r = np.sqrt(x**2+y**2+z**2)
        # theta = np.rad2deg(np.arctan(np.sqrt(x**2+z**2)/y))
        theta = np.rad2deg(np.arctan2(x, z))
        if theta >= -90 and theta <= 90:
            theta += 90
        elif theta >= -180 and theta < -90:
            theta += 90
        elif theta > 90 and theta <= 180:
            theta -= 270
        else:
            raise ValueError('theta out of range')
        # phi = np.rad2deg(np.arctan(z/x))+180
        phi = np.rad2deg(np.arccos(y/r))
        return [theta, phi, r, x, y, z]  # [:3] sperical cood, [3:] cartesian cood

    def hpose2camera(self, hpose):
        R = self.hpose2R(hpose)
        s = 1
        t3d = np.array([0., 0., 0.])
        R[:, :3] = R[:, :3] * s
        P = np.concatenate([R, t3d[:, None]], 1)
        P = np.concatenate([P, np.array([[0, 0, 0, 1.]])], 0)
        cam = self.fv_crpr.eg3dcamparams(P.flatten())
        return cam

    def estimate_rotation_angle(self, image_data, isBGR, box_np, pe, iterations=3):
        head_image, _ = self.crop_head_image(image_data.copy(), box_np)
        x1, y1, w, h = box_np
        x2, y2 = x1 + w, y1 + h
        box_quad = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]).astype(np.float32)
        box_center = np.mean(box_quad, axis=0)
        hpose = pe(head_image, isBGR=isBGR)
        R = self.hpose2R(hpose)
        angle = self.calculate_y_angle(R)
        for i in range(iterations):
            rot_img = self.rotate_image(image_data.copy(), box_center, angle)[0]
            rot_head_img, _ = self.crop_head_image(rot_img, box_np)
            hpose = pe(rot_head_img, isBGR=isBGR)
            R = self.hpose2R(hpose)
            angle += self.calculate_y_angle(R)
        return angle, box_center

    def generate_results(self, rotated_image, rot_quad, box_np, head_image_size, rotated_image_par=None, rotated_image_msk=None):
        x1, y1 = int(rot_quad[0, 0]), int(rot_quad[0, 1])
        x2, y2 = int(rot_quad[2, 0]), int(rot_quad[2, 1])
        w, h = x2 - x1, y2 - y1
        w = h = max(w, h)  # assert rot_quad is square
        head_crop_box = np.array([x1 - box_np[0], y1 - box_np[1], w, h])
        head_rot_quad = rot_quad - np.array([box_np[0], box_np[1]])
        head_image, head_pad_mask = self.crop_head_image(rotated_image.copy(), box_np)
        assert head_image.shape[0] == head_image.shape[1]
        scale = head_image_size / head_image.shape[0]
        head_image = cv2.resize(head_image, (head_image_size, head_image_size))
        head_pad_mask = cv2.resize(head_pad_mask, (head_image_size, head_image_size), interpolation=cv2.INTER_NEAREST)
        if rotated_image_par is not None:
            head_image_par = self.crop_head_parsing(rotated_image_par.copy(), box_np)
            head_image_par = cv2.resize(head_image_par, (head_image_size, head_image_size), interpolation=cv2.INTER_NEAREST)
        else:
            head_image_par = None
        if rotated_image_msk is not None:
            head_image_msk = self.crop_head_parsing(rotated_image_msk.copy(), box_np)
            head_image_msk = cv2.resize(head_image_msk, (head_image_size, head_image_size), interpolation=cv2.INTER_NEAREST)
        else:
            head_image_msk = None
        head_crop_box = head_crop_box * scale
        head_rot_quad = head_rot_quad * scale
        return head_image, head_crop_box, head_rot_quad, head_image_par, head_image_msk, head_pad_mask

    def process_back_view(self, img_data, img_msk_data, img_par_data, box_np):
        # Rotate image to make head vertical
        rot_quad = self.box2quad(self.transform_box(box_np.tolist(), self.mean_q2b_scale, self.mean_q2b_shift))
        rot_angle, rot_center = self.estimate_rotation_angle(img_data.copy(), isBGR=False, box_np=box_np,
                                                             pe=self.hed_pe, iterations=3)
        rotated_image, rotmat = self.rotate_image(img_data.copy(), rot_center, rot_angle)
        if img_par_data is not None:
            rotated_image_par, _ = self.rotate_parsing(img_par_data.copy(), rot_center, rot_angle)
        else:
            rotated_image_par = None
        if img_msk_data is not None:
            rotated_image_msk, _ = self.rotate_parsing(img_msk_data.copy(), rot_center, rot_angle)
        else:
            rotated_image_msk = None
        quad = self.rotate_quad(rot_quad, rot_center, -rot_angle)

        # Crop image
        cropped_img, _, tf_quad, none_padding_ratio, cropped_pad_mask = self.fv_crpr.crop_final(
            img_data.copy(), size=self.crop_size, quad=quad,
            top_expand=0., left_expand=0., bottom_expand=0., right_expand=0.
        )
        if img_par_data is not None:
            cropped_img_par, _, _ = self.fv_crpr.crop_final_parsing(
                img_par_data.copy(), size=self.crop_size, quad=quad,
                top_expand=0., left_expand=0., bottom_expand=0., right_expand=0.
            )
        else:
            cropped_img_par = None
        if img_msk_data is not None:
            cropped_img_msk, _, _ = self.fv_crpr.crop_final_parsing(
                img_msk_data.copy(), size=self.crop_size, quad=quad,
                top_expand=0., left_expand=0., bottom_expand=0., right_expand=0.
            )
        else:
            cropped_img_msk = None

        # Generate results
        head_image, head_crop_box, head_rot_quad, head_image_par, head_image_msk, head_pad_mask = self.generate_results(rotated_image, rot_quad, box_np, self.head_image_size, rotated_image_par, rotated_image_msk)
        assert head_image_par is not None
        assert head_image_msk is not None
        # if head_image_par is None:
        #     head_image_par = self.hed_par(head_image, isBGR=False, show=False)
        # if head_image_msk is None:
        #     head_image_msk = self.hed_seg(head_image, isBGR=False, show=False)

        if cropped_img_par is None:
            cropped_img_par = self.crop_head_parsing(head_image_par.copy(), head_crop_box)
            cropped_img_par = cv2.resize(cropped_img_par, (cropped_img.shape[1], cropped_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if cropped_img_msk is None:
            cropped_img_msk = self.crop_head_parsing(head_image_msk.copy(), head_crop_box)
            cropped_img_msk = cv2.resize(cropped_img_msk, (cropped_img.shape[1], cropped_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Estimate head pose and camera poses
        hpose = self.hed_pe(head_image, isBGR=False)
        camera_poses = self.hpose2camera(hpose)

        # Calculate q2b scale and shift
        quad_w = np.linalg.norm(quad[2] - quad[1])
        quad_h = np.linalg.norm(quad[1] - quad[0])
        quad_center = np.mean(quad, axis=0)
        hbox_w = box_np[2]
        hbox_h = box_np[3]
        hbox_center = box_np[:2] + box_np[2:] / 2.
        q2b_scale = [quad_w / hbox_w, quad_h / hbox_h]
        q2b_shift = [(quad_center[0] - hbox_center[0]) / hbox_w, (quad_center[1] - hbox_center[1]) / hbox_h]

        # Process results
        info_dict = {
            'raw': {
                'landmarks': None,
                'rotmat': rotmat.tolist(),  # Relative to raw image
                'rot_quad': rot_quad.tolist(),  # Relative to raw image
                'raw_quad': quad.tolist(),  # Relative to raw image
                'tgt_quad': tf_quad.tolist(),  # Relative to raw image
                'q2b_scale': q2b_scale,
                'q2b_shift': q2b_shift,
            },
            'head': {
                'align_box': head_crop_box.tolist(),  # Relative to cropped head image
                'align_quad': head_rot_quad.tolist(),  # Relative to cropped head image
                'hpose': hpose.astype(np.float32).tolist(),  # yaw, roll, pitch
                'camera': camera_poses.tolist(),  # 16 + 9
                'valid_area_ratio': none_padding_ratio,  # Valid Area / Head Box. Valid Area: area which is not the padding area when cropping. There may be area outside the image.
            }
        }
        #! head_pad_mask, cropped_pad_mask
        return info_dict, head_image, head_image_par, head_image_msk, head_pad_mask, cropped_img, cropped_img_par, cropped_img_msk, cropped_pad_mask

    def process_front_view(self, img_data, img_msk_data, img_par_data, box_np, landmarks):
        assert np.sum(landmarks < 0) == 0

        # Crop image
        cropped_img, camera_poses, quad, tf_quad, none_padding_ratio, cropped_pad_mask = self.fv_crpr(img_data.copy(), landmarks)
        quad, tf_quad = np.array(quad), np.array(tf_quad)

        # cam2world
        cam2world = camera_poses[:16].reshape(4, 4)
        P = np.linalg.inv(cam2world)
        R_in = self.inv_convert @ P
        R_in = R_in[:3, :3]
        hpose = self.R2hpose(R_in)

        # Calculate q2b scale and shift
        quad_w = np.linalg.norm(quad[2] - quad[1])
        quad_h = np.linalg.norm(quad[1] - quad[0])
        quad_center = np.mean(quad, axis=0)
        hbox_w = box_np[2]
        hbox_h = box_np[3]
        hbox_center = box_np[:2] + box_np[2:] / 2.
        q2b_scale = [quad_w / hbox_w, quad_h / hbox_h],
        q2b_shift = [(quad_center[0] - hbox_center[0]) / hbox_w, (quad_center[1] - hbox_center[1]) / hbox_h]

        # Rotate image
        rotated_image, rotmat, rot_quad = self.rotate_image_and_quad(img_data.copy(), quad, tf_quad,
                                                                     borderMode=cv2.BORDER_REFLECT, upsample=2)
        if img_par_data is not None or img_msk_data is not None:
            rot_mat_temp = cv2.getAffineTransform(tf_quad[:3], quad[:3])
            rot_angle = np.degrees(np.arctan2(rot_mat_temp[1, 0], rot_mat_temp[0, 0]))
            rot_center = np.mean(quad[:3], axis=0)

        if img_par_data is not None:
            rotated_image_par, _ = self.rotate_parsing(img_par_data.copy(), rot_center, rot_angle)
        else:
            rotated_image_par = None
        if img_msk_data is not None:
            rotated_image_msk, _ = self.rotate_parsing(img_msk_data.copy(), rot_center, rot_angle)
        else:
            rotated_image_msk = None

        # Generate results
        head_image, head_crop_box, head_rot_quad, head_image_par, head_image_msk, head_pad_mask = self.generate_results(rotated_image, rot_quad, box_np, self.head_image_size, rotated_image_par, rotated_image_msk)
        assert head_image_par is not None
        assert head_image_msk is not None
        # if head_image_par is None:
        #     head_image_par = self.hed_par(head_image, isBGR=False, show=False)
        # if head_image_msk is None:
        #     head_image_msk = self.hed_seg(head_image, isBGR=False, show=False)
        cropped_img_par = self.crop_head_parsing(head_image_par.copy(), head_crop_box)
        cropped_img_par = cv2.resize(cropped_img_par, (cropped_img.shape[1], cropped_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cropped_img_msk = self.crop_head_parsing(head_image_msk.copy(), head_crop_box)
        cropped_img_msk = cv2.resize(cropped_img_msk, (cropped_img.shape[1], cropped_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Process results
        info_dict = {
            'raw': {
                'landmarks': landmarks.tolist(),
                'rotmat': rotmat.tolist(),  # Relative to raw image
                'rot_quad': rot_quad.tolist(),  # Relative to raw image
                'raw_quad': quad.tolist(),  # Relative to raw image
                'tgt_quad': tf_quad.tolist(),  # Relative to raw image
                'q2b_scale': q2b_scale,
                'q2b_shift': q2b_shift,
            },
            'head': {
                'align_box': head_crop_box.tolist(),  # Relative to cropped head image
                'align_quad': head_rot_quad.tolist(),  # Relative to cropped head image
                'hpose': hpose,  # yaw, roll, pitch
                'camera': camera_poses.tolist(),  # 16 + 9
                'valid_area_ratio': none_padding_ratio,  # Valid Area / Head Box. Valid Area: area which is not the padding area when cropping. There may be area outside the image.
            }
        }
        #! head_pad_mask, cropped_pad_mask
        return info_dict, head_image, head_image_par, head_image_msk, head_pad_mask, cropped_img, cropped_img_par, cropped_img_msk, cropped_pad_mask
