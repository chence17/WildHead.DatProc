import cv2
import torch
import onnxruntime
import numpy as np
import numba as nb


class HeadDetector(object):
    def __init__(self, weights_file: str, input_width: int = 640, input_height: int = 480, conf_thresh: float = 0.60,
                 nms_thresh: float = 0.50, size_thres: int = 512,
                 sort_by_wh: bool = True) -> None:
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
        assert self.output_shapes[0] == [1, 18900, 1, 4]  # boxes[N, num, classes, boxes]
        assert self.output_shapes[1] == [1, 18900, 1]  # confs[N, num, classes]
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.size_thres = size_thres
        self.sort_by_wh = sort_by_wh

    def __repr__(self) -> str:
        return f'HeadDetector(weights_file={self.weights_file}, input_width={self.input_width}, input_height={self.input_height}, conf_thresh={self.conf_thresh}, nms_thresh={self.nms_thresh}, size_thres={self.size_thres})'

    def __call__(self, image_data: np.ndarray, isBGR: bool, max_box_num=1) -> np.ndarray:
        """Detect head boxes from image data.

        Args:
            image_data (np.ndarray): image data
            isBGR (bool): BGR or RGB
            max_box_num (int, optional): Maximum number of saved boxes. Defaults to 3.

        Returns:
            np.ndarray: Detected boxes. N * 4, each box is [x_min, y_min, w, h].
        """
        if isBGR:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image_data_padded, pad_list = self.resize_and_pad_image(image_data, self.input_hw, pad_value=0)
        image_data_chw = image_data_padded.transpose(2, 0, 1).astype(np.float32) / 255.
        image_data_boxes, image_data_confs = self.detector.run(
            output_names=self.output_names, input_feed={self.input_name: image_data_chw[np.newaxis, ...]}
        )
        image_data_boxes, image_data_confs = image_data_boxes[0][:, 0, :], image_data_confs[0][:, 0]
        argwhere = image_data_confs > self.conf_thresh
        image_data_boxes, image_data_confs = image_data_boxes[argwhere, :], image_data_confs[argwhere]
        image_data_heads = []
        image_data_keep = self.nms_cpu(
            boxes=image_data_boxes, confs=image_data_confs, nms_thresh=self.nms_thresh, min_mode=False
        )
        if image_data_keep.size == 0:
            return None
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
            cur_head_box = self.recover_original_box(image_data_heads[idx], pad_list, original_hw)
            if self.sort_by_wh:
                cur_head_box[4] *= cur_head_box[2] * cur_head_box[3]
            image_data_heads_.append(cur_head_box)
        scaled_boxes = np.apply_along_axis(self.rescale_headbox, 1, image_data_heads_, image_data.shape[1],
                                           image_data.shape[0])
        filtered_boxes = scaled_boxes[(scaled_boxes[:, 3] >= self.size_thres) & (scaled_boxes[:, 2] >= self.size_thres)]
        sorted_indices = np.argsort(filtered_boxes[:, 4])[::-1]
        filtered_boxes = filtered_boxes[sorted_indices]
        return filtered_boxes[:max_box_num, :4].astype(np.int32)

    @staticmethod
    def rescale_headbox(box, image_w, image_h, factor=1.2, allow_overflow=True):
        # expected input: [x_min, y_min, w, h]
        # image_w, image_h: original image size
        # return [x_min, y_min, w, h, w*h], w = h, square
        if not allow_overflow:
            size = max(box[2], box[3])
            x_min = max(box[0] - (factor - 1) * size / 2, 0)
            y_min = max(box[1] - (factor - 1) * size / 2, 0)
            x_max = min(box[0] + size + (factor - 1) * size / 2, image_w)
            y_max = min(box[1] + size + (factor - 1) * size / 2, image_h)
            w = x_max - x_min
            h = y_max - y_min
        else:
            x_min, y_min, w, h = box[0], box[1], box[2], box[3]
            size = max(w, h)
            cx, cy = x_min + w/2., y_min + h/2.
            w, h = size * factor, size * factor
            x_min, y_min = cx - w/2., cy - h/2.
        return np.array([x_min, y_min, w, h, w * h]).astype(np.float32)

    @staticmethod
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

    @staticmethod
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
