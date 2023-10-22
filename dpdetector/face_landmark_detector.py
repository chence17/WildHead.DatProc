'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-10-15 21:06:25
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-10-15 21:07:21
FilePath: /DatProc/detector/face_landmark_detector.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import dlib
import torch
import numpy as np
import face_alignment


class FaceAlignmentDetector():
    def __init__(self, score_thres=0.75) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False,
                                                     device=device, face_detector='sfd')
        self.score_thres = score_thres

    def __call__(self, image_data: np.ndarray, isBGR: bool, image_upper_left=None):
        # The image_data here is a cropped region from original image.
        # The output of this function is the absolute coorinate of landmarks in the image
        if isBGR: image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        max_dim = np.argmax(image_data.shape[:2])
        scale = 512 / image_data.shape[max_dim]
        image_data = cv2.resize(image_data, (int(image_data.shape[1] * scale), int(image_data.shape[0] * scale)))
        landmarks = self.detector.get_landmarks_from_image(image_data, return_landmark_score=True)
        if landmarks[0] is None:
            return None
        else:
            scores = np.array(landmarks[1])
            scores = np.mean(scores, axis=1)
            max_score = np.max(scores)
            max_score_idx = np.argmax(scores)
            if max_score < self.score_thres:
                return None
            ret_landmarks = landmarks[0][max_score_idx][:, :2] * (1 / scale)
        if image_upper_left is not None:
            return ret_landmarks + image_upper_left
        else:
            return ret_landmarks


class DlibDetector():
    def __init__(self,
                 weight_file="/home/shitianhao/project/DatProc/3DDFA_V2/weights/shape_predictor_68_face_landmarks.dat",
                 score_thres=0.85) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(weight_file)
        self.score_thres = score_thres

    def __call__(self, image_data: np.ndarray, isBGR: bool, image_upper_right: np.array):
        # The image_data here is a cropped region from original image.
        # The output of this function is the absolute coorinate of landmarks in the image
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY) if isBGR else cv2.cvtColor(image_data,
                                                                                             cv2.COLOR_RGB2GRAY)
        rects = self.detector(image_data, 1)  # detect face
        if len(rects) == 0: return None
        # since dlib doesn't provide API to measure the confidence of face detection, we use the first face detected
        shape = self.predictor(image_data, rects[0])
        landmarks = [np.array([p.x, p.y]) + image_upper_right for p in shape.parts()]
        return landmarks
