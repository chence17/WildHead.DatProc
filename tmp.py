'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-10-16 13:48:12
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-10-17 17:37:15
FilePath: /DatProc/tmp.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import cv2

from scipy.spatial.transform import Rotation

from dpfilter import ImageSizeFilter, ImageBlurFilter
from dpdetector import HeadDetector, FaceAlignmentDetector
from dpcropper import FrontViewCropper
from dpparser import HeadParser
from dpestimator import HeadPoseEstimator


class DatProcV1(object):
    def __init__(self) -> None:
        self.img_sz_flt = ImageSizeFilter(size_thres=512)
        self.img_br_flt = ImageBlurFilter(svd_thres=0.6, lap_thres=100)
        self.hed_det = HeadDetector(weights_file='assets/224x224_yolov4_hddet_480x640.onnx', input_width=640,
                                    input_height=480, size_thres=512)
        self.flmk_det = FaceAlignmentDetector(score_thres=0.8)
        self.fv_crpr = FrontViewCropper(config_file='TDDFA_V2/configs/mb1_120x120.yml', mode='gpu')
        self.hed_par = HeadParser()
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
    def crop_head_image(image_data, box):
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
            crop_msk = np.ones_like(crop_img) * 255
            crop_img = cv2.copyMakeBorder(crop_img, top_size, bottom_size, left_size, right_size,
                                          borderType=cv2.BORDER_REFLECT)
            crop_msk = cv2.copyMakeBorder(crop_msk, top_size, bottom_size, left_size, right_size,
                                          borderType=cv2.BORDER_CONSTANT, value=0)
            size = max(crop_img.shape[:2])
            mask_kernel = int(size*0.02)*2+1
            blur_kernel = int(size*0.03)*2+1
            blur_mask = cv2.blur(crop_msk.astype(np.float32).mean(2), (mask_kernel, mask_kernel)) / 255.0
            blur_mask = blur_mask[..., np.newaxis]  # .astype(np.float32) / 255.0
            blurred_img = cv2.blur(crop_img, (blur_kernel, blur_kernel), 0)
            crop_img = crop_img * blur_mask + blurred_img * (1 - blur_mask)
            crop_img = crop_img.astype(np.uint8)
        # print(crop_img.shape, (h, w))
        assert crop_img.shape[0] == h
        assert crop_img.shape[1] == w
        return crop_img

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
            crop_img = cv2.copyMakeBorder(crop_img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT, value=bg_value)
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
            crop_img = cv2.warpAffine(np.array(img), upsample * rotmat, crop_size_large,
                                    flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
            crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_AREA)

        empty = np.ones_like(img) * 255
        crop_mask = cv2.warpAffine(empty, rotmat, crop_size)

        size = min(crop_w, crop_h)
        mask_kernel = int(size*0.02)*2+1
        blur_kernel = int(size*0.03)*2+1

        if crop_mask.mean() < 255:
            blur_mask = cv2.blur(crop_mask.astype(np.float32).mean(2), (mask_kernel, mask_kernel)) / 255.0
            blur_mask = blur_mask[..., np.newaxis]  #.astype(np.float32) / 255.0
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
        head_image = self.crop_head_image(image_data.copy(), box_np)
        x1, y1, w, h = box_np
        x2, y2 = x1 + w, y1 + h
        box_quad = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]).astype(np.float32)
        box_center = np.mean(box_quad, axis=0)
        hpose = pe(head_image, isBGR=isBGR)
        R = self.hpose2R(hpose)
        angle = self.calculate_y_angle(R)
        for i in range(iterations):
            rot_img = self.rotate_image(image_data.copy(), box_center, angle)[0]
            rot_head_img = self.crop_head_image(rot_img, box_np)
            hpose = pe(rot_head_img, isBGR=isBGR)
            R = self.hpose2R(hpose)
            angle += self.calculate_y_angle(R)
        return angle, box_center

    def generate_results(self, rotated_image, rot_quad, box_np, head_image_size):
        x1, y1 = int(rot_quad[0, 0]), int(rot_quad[0, 1])
        x2, y2 = int(rot_quad[2, 0]), int(rot_quad[2, 1])
        w, h = x2 - x1, y2 - y1
        w = h = max(w, h)  # assert rot_quad is square
        head_crop_box = np.array([x1 - box_np[0], y1 - box_np[1], w, h])
        head_rot_quad = rot_quad - np.array([box_np[0], box_np[1]])
        head_image = self.crop_head_image(rotated_image.copy(), box_np)
        assert head_image.shape[0] == head_image.shape[1]
        scale = head_image_size / head_image.shape[0]
        head_image = cv2.resize(head_image, (head_image_size, head_image_size))
        head_crop_box = head_crop_box * scale
        head_rot_quad = head_rot_quad * scale
        return head_image, head_crop_box, head_rot_quad

    def process_back_view(self, img_data, box_np):
        # Rotate image to make head vertical
        rot_quad = self.box2quad(self.transform_box(box_np.tolist(), self.mean_q2b_scale, self.mean_q2b_shift))
        rot_angle, rot_center = self.estimate_rotation_angle(img_data.copy(), isBGR=False, box_np=box_np,
                                                             pe=self.hed_pe, iterations=3)
        rotated_image, rotmat = self.rotate_image(img_data.copy(), rot_center, rot_angle)
        quad = self.rotate_quad(rot_quad, rot_center, -rot_angle)

        # Crop image
        cropped_img, _, tf_quad = self.fv_crpr.crop_final(
            img_data.copy(), size=self.crop_size, quad=quad,
            top_expand=0., left_expand=0., bottom_expand=0., right_expand=0.
        )

        # Generate results
        head_image, head_crop_box, head_rot_quad = self.generate_results(rotated_image, rot_quad, box_np,
                                                                         self.head_image_size)
        head_parsing = self.hed_par(head_image, isBGR=False, show=False)
        cropped_par = self.crop_head_parsing(head_parsing.copy(), head_crop_box)
        cropped_par = cv2.resize(cropped_par, (cropped_img.shape[1], cropped_img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

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
            }
        }

        return info_dict, head_image, head_parsing, cropped_img, cropped_par

    def process_front_view(self, img_data, box_np, landmarks):
        assert np.sum(landmarks < 0) == 0

        # Crop image
        cropped_img, camera_poses, quad, tf_quad = self.fv_crpr(img_data.copy(), landmarks)
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

        # Generate results
        head_image, head_crop_box, head_rot_quad = self.generate_results(rotated_image, rot_quad, box_np,
                                                                         self.head_image_size)
        head_parsing = self.hed_par(head_image, isBGR=False, show=False)
        cropped_par = self.crop_head_parsing(head_parsing.copy(), head_crop_box)
        cropped_par = cv2.resize(cropped_par, (cropped_img.shape[1], cropped_img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

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
            }
        }

        return info_dict, head_image, head_parsing, cropped_img, cropped_par
