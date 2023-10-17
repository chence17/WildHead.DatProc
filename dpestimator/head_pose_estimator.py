'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-10-17 16:23:57
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-10-17 16:25:00
FilePath: /DatProc/dpestimator/head_pose_estimator.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np
import torch
import onnxruntime


class HeadPoseEstimator(object):
    """WHENetHeadPoseEstimator

    Args:
        object (_type_): _description_
    """
    def __init__(self, weights_file: str, input_width: int = 224, input_height: int = 224) -> None:
        self.weights_file = weights_file
        self.providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            self.providers.insert(0, 'CUDAExecutionProvider')
        self.estimator = onnxruntime.InferenceSession(self.weights_file, providers=self.providers)
        self.input_width = input_width
        self.input_height = input_height
        self.input_hw = (self.input_height, self.input_width)
        self.input_name = self.estimator.get_inputs()[0].name
        self.output_names = [output.name for output in self.estimator.get_outputs()]
        self.output_shapes = [output.shape for output in self.estimator.get_outputs()]

    def __call__(self, image_data: np.ndarray, isBGR: bool) -> np.ndarray:
        if isBGR:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        h, w = image_data.shape[:2]
        if (h, w) != self.input_hw:
            image_data = self.resize_and_pad_image(image_data, self.input_hw, border_type=cv2.BORDER_REFLECT)[0]
        chw = image_data.transpose(2, 0, 1)
        nchw = np.asarray(chw[np.newaxis, :, :, :], dtype=np.float32)
        outputs = self.estimator.run(
            output_names=self.output_names,
            input_feed={self.input_name: nchw}
        )
        yaw, roll, pitch = outputs[0][0][0], outputs[0][0][1], outputs[0][0][2]
        yaw, roll, pitch = np.squeeze([yaw, roll, pitch])
        return np.array([yaw, roll, pitch])

    @staticmethod
    def resize_and_pad_image(image_data, target_hw, border_type=cv2.BORDER_CONSTANT, pad_value=0):
        image_hw = image_data.shape[:2]
        image_scale = min(target_hw[0] / image_hw[0], target_hw[1] / image_hw[1])
        new_h = int(image_hw[0] * image_scale)
        new_w = int(image_hw[1] * image_scale)
        image_data_scaled = cv2.resize(image_data, (new_w, new_h))
        pad_top = (target_hw[0] - new_h) // 2
        pad_bottom = target_hw[0] - new_h - pad_top
        pad_left = (target_hw[1] - new_w) // 2
        pad_right = target_hw[1] - new_w - pad_left
        image_data_padded = cv2.copyMakeBorder(image_data_scaled, pad_top, pad_bottom, pad_left, pad_right,
                                               borderType=border_type, value=pad_value)
        if border_type != cv2.BORDER_CONSTANT:
            image_mask_scaled = np.ones_like(image_data_scaled) * 255
            image_mask_padded = cv2.copyMakeBorder(image_mask_scaled, pad_top, pad_bottom, pad_left, pad_right,
                                                   borderType=cv2.BORDER_CONSTANT, value=0)
            size = max(image_data_padded.shape[:2])
            mask_kernel = int(size*0.02)*2+1
            blur_kernel = int(size*0.03)*2+1
            blur_mask = cv2.blur(image_mask_padded.astype(np.float32).mean(2), (mask_kernel, mask_kernel)) / 255.0
            blur_mask = blur_mask[..., np.newaxis]  # .astype(np.float32) / 255.0
            blurred_img = cv2.blur(image_data_padded, (blur_kernel, blur_kernel), 0)
            image_data_padded = image_data_padded * blur_mask + blurred_img * (1 - blur_mask)
            image_data_padded = image_data_padded.astype(np.uint8)
        assert image_data_padded.shape[:2] == target_hw
        pad_list = np.array([pad_top, pad_bottom, pad_left, pad_right, image_scale]).astype(np.float32)
        return image_data_padded, pad_list
