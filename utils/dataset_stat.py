import os
import cv2
import dlib
import json
import torch
import argparse
import onnxruntime
import numpy as np
import numba as nb
import face_alignment
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Statistic')
    parser.add_argument('--data_path', type=str, help='path to dataset', default='/home/shitianhao/project/DatProc/assets/mh_dataset')
    parser.add_argument('--output_dir', type=str, help='path to output dir', default='/home/shitianhao/project/DatProc/utils/stats')
    args, _ = parser.parse_known_args()
    return args

def get_all_extensions(data_path):
    extensions = set()
    for root, dirs, files in os.walk(data_path):
        for file in files:
            ext = os.path.splitext(file)[-1]
            extensions.add(ext)
    return extensions

def get_images(data_path, target_file_exts=['.jpg', '.png']):
    img_paths = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            ext = os.path.splitext(file)[-1]
            if ext in target_file_exts:
                photo_path = os.path.join(root, file)
                img_paths.append(photo_path)
    return img_paths

@nb.njit('int64[:](float32[:,:], float32[:], float32, bool_)', fastmath=True, cache=True)
def nms_cpu(boxes, confs, nms_thresh, min_mode):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]
    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]
        keep.append(idx_self)
        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)
        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    return np.array(keep).astype(np.int64)

def resize_and_pad_image(image_data, target_hw, pad_value=0):
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
                                         borderType=cv2.BORDER_CONSTANT, value=pad_value)
    assert image_data_padded.shape[:2] == target_hw
    pad_list = np.array([pad_top, pad_bottom, pad_left, pad_right, image_scale]).astype(np.float32)
    return image_data_padded, pad_list

def recover_original_box(head_box, pad_list, original_hw):
    pad_top, pad_bottom, pad_left, pad_right, image_scale = pad_list
    x_min = (head_box[0] - pad_left) / image_scale
    y_min = (head_box[1] - pad_top) / image_scale
    x_max = (head_box[2] - pad_left) / image_scale
    y_max = (head_box[3] - pad_top) / image_scale
    # enlarge the bbox to include more background margin
    y_min = max(0, y_min - abs(y_min - y_max) / 10)
    y_max = min(original_hw[0], y_max + abs(y_min - y_max) / 10)
    y_center = (y_min + y_max) / 2
    y_delta = (y_max - y_min) / 2
    x_min = max(0, x_min - abs(x_min - x_max) / 5)
    x_max = min(original_hw[1], x_max + abs(x_min - x_max) / 5)
    x_max = min(x_max, original_hw[1])
    x_center = (x_min + x_max) / 2
    x_delta = (x_max - x_min) / 2
    xy_delta = max(x_delta, y_delta)
    y_min = max(0, y_center - xy_delta)
    y_max = min(original_hw[0], y_center + xy_delta)
    x_min = max(0, x_center - xy_delta)
    x_max = min(original_hw[1], x_center + xy_delta)
    image_head_width = (x_max - x_min)
    image_head_height = (y_max - y_min)
    image_head_width = max(image_head_width, image_head_height)
    image_head_height = image_head_width
    image_head_cx = ((x_min + x_max) / 2.)
    image_head_cy = ((y_min + y_max) / 2.)
    x1, y1 = image_head_cx - image_head_width / 2, image_head_cy - image_head_height / 2
    x2, y2 = image_head_cx + image_head_width / 2, image_head_cy + image_head_height / 2
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(original_hw[1], x2), min(original_hw[0], y2)
    half_w = min(abs(x1 - image_head_cx), abs(x2 - image_head_cx))
    half_h = min(abs(y1 - image_head_cy), abs(y2 - image_head_cy))
    image_head_width, image_head_height = 2. * half_w, 2. * half_h
    return np.array([x1, y1, image_head_width, image_head_height, head_box[4]]).astype(np.float32)

def rescale_headbox(box, image_w, image_h, factor=1.2):
    # expected input: [x_min, y_min, w, h]
    # image_w, image_h: original image size
    # return [x_min, y_min, w, h, w*h]
    x_min = max(box[0] - (factor - 1) * box[2] / 2, 0)
    y_min = max(box[1] - (factor - 1) * box[3] / 2, 0)
    x_max = min(box[0] + box[2] + (factor - 1) * box[2] / 2, image_w)
    y_max = min(box[1] + box[3] + (factor - 1) * box[3] / 2, image_h)
    w = x_max - x_min
    h = y_max - y_min
    return np.array([x_min, y_min, w, h, w*h]).astype(np.float32)



class YoloHeadDetector(object):
    def __init__(self, weights_file: str, input_width: int=640, input_height: int=480) -> None:
        self.weights_file = weights_file
        self.providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            self.providers.insert(0, 'CUDAExecutionProvider')
        self.detector = onnxruntime.InferenceSession(self.weights_file, providers=self.providers)
        self.input_width = input_width
        self.input_height = input_height
        self.input_hw = (self.input_height, self.input_width)
        self.input_name = self.detector.get_inputs()[0].name
        self.output_names = [output.name for output in self.detector.get_outputs()]
        self.output_shapes = [output.shape for output in self.detector.get_outputs()]
        assert self.output_shapes[0] == [1, 18900, 1, 4] # boxes[N, num, classes, boxes]
        assert self.output_shapes[1] == [1, 18900, 1]    # confs[N, num, classes]
        self.conf_thresh = 0.60
        self.nms_thresh = 0.50


    def __call__(self, image_data: np.ndarray, isBGR: bool, max_box_num=3) -> np.ndarray:
        if isBGR: image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image_data_padded, pad_list = resize_and_pad_image(image_data, self.input_hw, pad_value=0)
        image_data_chw = image_data_padded.transpose(2, 0, 1).astype(np.float32) / 255.
        image_data_boxes, image_data_confs = self.detector.run(
            output_names=self.output_names, input_feed={self.input_name: image_data_chw[np.newaxis, ...]}
        )
        image_data_boxes, image_data_confs = image_data_boxes[0][:, 0, :], image_data_confs[0][:, 0]
        argwhere = image_data_confs > self.conf_thresh
        image_data_boxes, image_data_confs = image_data_boxes[argwhere, :], image_data_confs[argwhere]
        image_data_heads = []
        image_data_keep = nms_cpu(
            boxes=image_data_boxes, confs=image_data_confs, nms_thresh=self.nms_thresh, min_mode=False
        )
        if image_data_keep.size == 0: print('No object detected!'); return None
        width = image_data_padded.shape[1]
        height = image_data_padded.shape[0]
        if (image_data_keep.size > 0):
            image_data_boxes, image_data_confs = image_data_boxes[image_data_keep, :], image_data_confs[image_data_keep]
            for k in range(image_data_boxes.shape[0]):
                image_data_heads.append([image_data_boxes[k, 0] * width, image_data_boxes[k, 1] * height,
                                       image_data_boxes[k, 2] * width, image_data_boxes[k, 3] * height,
                                       image_data_confs[k]])
        original_hw, image_data_heads_ = image_data.shape[:2], []
        for idx in range(len(image_data_heads)):
            image_data_heads_.append(recover_original_box(image_data_heads[idx], pad_list, original_hw))
        scaled_boxes = np.apply_along_axis(rescale_headbox, 1, image_data_heads_, image_data.shape[1], image_data.shape[0])
        filtered_boxes = scaled_boxes[(scaled_boxes[:, 3] >= 512) & (scaled_boxes[:, 2] >= 512)]
        sorted_indices = np.argsort(filtered_boxes[:, 4])[::-1]
        filtered_boxes = filtered_boxes[sorted_indices]
        return filtered_boxes[:max_box_num, :4].astype(np.int32)

class FaceAlignmentDetector():
    def __init__(self, score_thres=0.8) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device=device, face_detector='sfd')
        self.score_thres = score_thres
    
    def __call__(self, image_data: np.ndarray, isBGR: bool, image_upper_right:np.array) -> np.ndarray:
        # The image_data here is a cropped region from original image.
        # The output of this function is the absolute coorinate of landmarks in the image
        if isBGR: image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        landmarks = self.detector.get_landmarks_from_image(image_data, return_landmark_score=True)
        if landmarks[0] is None:
            print("No face detected!")
            return None
        else:
            scores = np.array(landmarks[1])
            scores = np.mean(scores, axis=1)
            max_score = np.max(scores)
            max_score_idx = np.argmax(scores)
            print(max_score)
            if max_score < self.score_thres:
                print("Score is too low!")
                return None
        return landmarks[0][max_score_idx][:, :2] + image_upper_right if image_upper_right is not None else landmarks[0][max_score_idx][:, :2]

class DlibDetector():
    def __init__(self, weight_file="/home/shitianhao/project/DatProc/3DDFA_V2/weights/shape_predictor_68_face_landmarks.dat", score_thres=0.85) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(weight_file)
        self.score_thres = score_thres
    
    def __call__(self, image_data: np.ndarray, isBGR: bool, image_upper_right:np.array) -> np.ndarray:
        # The image_data here is a cropped region from original image.
        # The output of this function is the absolute coorinate of landmarks in the image
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY) if isBGR else cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
        rects = self.detector(image_data, 1) # detect face
        if len(rects) == 0: return None
        # since dlib doesn't provide API to measure the confidence of face detection, we use the first face detected
        shape = self.predictor(image_data, rects[0])
        landmarks = [np.array([p.x, p.y]) + image_upper_right for p in shape.parts()]
        return landmarks

if __name__ == '__main__':
    hdet = YoloHeadDetector(weights_file='assets/224x224_yolov4_hddet_480x640.onnx',
                            input_width=640, input_height=480)
    
    img = cv2.imread("/home/shitianhao/project/DatProc/assets/mh_dataset/5.png")
    boxes = hdet(img, isBGR=True)