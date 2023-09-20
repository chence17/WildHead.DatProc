import cv2
import numpy as np
import torch
import onnxruntime
from utils.head_detection import resize_and_pad_image

class WHENetHeadPoseEstimator(object):
    def __init__(self, weights_file: str, input_width: int=224, input_height: int=224) -> None:
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
            image_data = resize_and_pad_image(image_data, self.input_hw, border_type=cv2.BORDER_REFLECT)[0]
        chw = image_data.transpose(2, 0, 1)
        nchw = np.asarray(chw[np.newaxis, :, :, :], dtype=np.float32)
        outputs = self.estimator.run(
            output_names=self.output_names,
            input_feed={self.input_name: nchw}
        )
        yaw, roll, pitch = outputs[0][0][0], outputs[0][0][1], outputs[0][0][2]
        yaw, roll, pitch = np.squeeze([yaw, roll, pitch])
        return np.array([yaw, roll, pitch])
