'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-03-02 13:19:19
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-05-25 17:41:56
FilePath: /DataProcess/utils/mface.py
Description: mface.py
'''
import os.path as osp
import os
import tqdm
import numpy as np
import cv2
import torch
import face_alignment
import copy
from skimage import measure
from matplotlib import pyplot as plt
from collections import namedtuple
from typing import Dict
from torchvision.transforms import transforms
from skimage.transform import SimilarityTransform

from utils import msimil
from utils.mdata import MFaceBBox, MFaceBBoxLandmarks, MVideoFrames, MHeadBBox, MFaceKeyPoints
from utils.mdata import MCamera, MTransMatrix
from utils.mvideo import crop_frame
from utils.mmesh import MRaysMesh
from bisenet.bisenet import load_BiSeNet_model
from PIL import Image
from ibug.face_parsing import FaceParser

ours_label_list = [
    'background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip',
    'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth'
]
ours_color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
                   [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
                   [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
ours_label2num = {label: i for i, label in enumerate(ours_label_list)}
ours_num2label = {i: label for i, label in enumerate(ours_label_list)}
bisenet_num2label = {
    0: 'background',  # 背景
    1: 'skin',  # 面部皮肤
    2: 'l_brow',  # 左眉毛，left_eyebrow
    3: 'r_brow',  # 右眉毛，right_eyebrow
    4: 'l_eye',  # 左眼，left_eye
    5: 'r_eye',  # 右眼，right_eye
    6: 'eye_g',  # 眼镜，eye_glasses
    7: 'l_ear',  # 左耳，left_ear
    8: 'r_ear',  # 右耳，right_ear
    9: 'ear_r',  # 耳环，ear_rings
    10: 'nose',  # 鼻子
    11: 'mouth',  # 嘴巴
    12: 'u_lip',  # 上嘴唇，upper_lip
    13: 'l_lip',  # 下嘴唇，lower_lip
    14: 'neck',  # 脖子
    15: 'neck_l',  # 项链，necklace
    16: 'cloth',  # 衣服
    17: 'hair',  # 头发
    18: 'hat'  # 帽子
}
bisenet_label2num = {label: i for i, label in bisenet_num2label.items()}
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)


def detect_face_bboxes(vfrm: MVideoFrames, model_file: str, score_threshold: float, nms_threshold: float, top_k: int):
    fd = cv2.FaceDetectorYN.create(model=model_file,
                                   config='',
                                   input_size=(vfrm.width, vfrm.height),
                                   score_threshold=score_threshold,
                                   nms_threshold=nms_threshold,
                                   top_k=top_k,
                                   backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
                                   target_id=cv2.dnn.DNN_TARGET_CPU)
    results = {}
    for frame in tqdm.tqdm(vfrm.frames, desc=f'Face Detection for {vfrm.path}'):
        frame_path = osp.join(vfrm.path, frame)
        frame_data = cv2.imread(frame_path)
        _, bboxes = fd.detect(frame_data)
        if bboxes is None or len(bboxes) == 0 or len(bboxes) >= 2:
            results[frame] = None
        else:
            top_left = np.array([bboxes[0][0], bboxes[0][1]])
            width = bboxes[0][2]
            height = bboxes[0][3]
            score = bboxes[0][-1]
            landmarks_np = bboxes[0][4:14].reshape((5, 2))
            landmarks = MFaceBBoxLandmarks(right_eye=landmarks_np[0],
                                           left_eye=landmarks_np[1],
                                           nose_tip=landmarks_np[2],
                                           right_mouth_corner=landmarks_np[3],
                                           left_mouth_corner=landmarks_np[4])
            results[frame] = MFaceBBox(top_left=top_left, width=width, height=height, score=score, landmarks=landmarks)
    return results


def detect_fv_face_bboxes(vfrm: MVideoFrames, fv_list: list, model_file: str, score_threshold: float,
                          nms_threshold: float, top_k: int):
    fd = cv2.FaceDetectorYN.create(model=model_file,
                                   config='',
                                   input_size=(vfrm.width, vfrm.height),
                                   score_threshold=score_threshold,
                                   nms_threshold=nms_threshold,
                                   top_k=top_k,
                                   backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
                                   target_id=cv2.dnn.DNN_TARGET_CPU)
    results = {}
    for frame in tqdm.tqdm(vfrm.frames, desc=f'Face Detection for {vfrm.path}'):
        if frame not in fv_list:
            results[frame] = None
            continue
        frame_path = osp.join(vfrm.path, frame)
        frame_data = cv2.imread(frame_path)
        _, bboxes = fd.detect(frame_data)
        if bboxes is None or len(bboxes) == 0 or len(bboxes) >= 2:
            results[frame] = None
        else:
            top_left = np.array([bboxes[0][0], bboxes[0][1]])
            width = bboxes[0][2]
            height = bboxes[0][3]
            score = bboxes[0][-1]
            landmarks_np = bboxes[0][4:14].reshape((5, 2))
            landmarks = MFaceBBoxLandmarks(right_eye=landmarks_np[0],
                                           left_eye=landmarks_np[1],
                                           nose_tip=landmarks_np[2],
                                           right_mouth_corner=landmarks_np[3],
                                           left_mouth_corner=landmarks_np[4])
            results[frame] = MFaceBBox(top_left=top_left, width=width, height=height, score=score, landmarks=landmarks)
    return results


def visualize_face_bboxes(vfrm: MVideoFrames,
                          face_bboxes: Dict[str, MFaceBBox],
                          save_dir: str,
                          thickness: int = 2,
                          font_scale: float = 0.5,
                          circle_radius: int = 2):
    os.makedirs(save_dir, exist_ok=True)
    bbox_color = (0, 255, 0)  # BGR
    text_color = (0, 0, 255)  # BGR
    landmark_color = [  # BGR
        (255, 0, 0),  # right eye
        (0, 0, 255),  # left eye
        (0, 255, 0),  # nose tip
        (255, 0, 255),  # right mouth corner
        (0, 255, 255)  # left mouth corner
    ]
    for frame in tqdm.tqdm(vfrm.frames, desc=f'Face Detection Visualization for {vfrm.path}'):
        frame_path = osp.join(vfrm.path, frame)
        frame_data = cv2.imread(frame_path)
        bbox = face_bboxes[frame]
        if bbox is not None:
            tl = bbox.top_left
            br = [tl[0] + bbox.width, tl[1] + bbox.height]
            tl = np.int32(tl).tolist()
            br = np.int32(br).tolist()
            cv2.rectangle(frame_data, tl, br, bbox_color, thickness)
            if bbox.score is not None:
                cv2.putText(frame_data, '{:.4f}'.format(bbox.score), (tl[0], tl[1] + 12), cv2.FONT_HERSHEY_DUPLEX,
                            font_scale, text_color)
            if bbox.landmarks is not None:
                landmark_points = [
                    np.int32(bbox.landmarks.right_eye).tolist(),
                    np.int32(bbox.landmarks.left_eye).tolist(),
                    np.int32(bbox.landmarks.nose_tip).tolist(),
                    np.int32(bbox.landmarks.right_mouth_corner).tolist(),
                    np.int32(bbox.landmarks.left_mouth_corner).tolist()
                ]
                cv2.circle(frame_data, landmark_points[0], circle_radius, landmark_color[0], thickness)
                cv2.circle(frame_data, landmark_points[1], circle_radius, landmark_color[1], thickness)
                cv2.circle(frame_data, landmark_points[2], circle_radius, landmark_color[2], thickness)
                cv2.circle(frame_data, landmark_points[3], circle_radius, landmark_color[3], thickness)
                cv2.circle(frame_data, landmark_points[4], circle_radius, landmark_color[4], thickness)
        cv2.imwrite(osp.join(save_dir, frame), frame_data)
    return True


def calculate_nearest_point_of_rays(rays_o: np.ndarray, rays_d: np.ndarray):
    # rays_o, rays_d: [N, 3, 1], sum(rays_d ** 2) == 1
    A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
    b_i = -A_i @ rays_o
    pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
    return pt_mindist


def calculate_head_center(fbboxes, cams, vis_path=None):
    head_centers2D = {}
    head_cameras = {}
    for k in fbboxes.keys():
        if fbboxes[k] is not None:
            bbox = fbboxes[k]
            tl = bbox.top_left
            center2D = [tl[0] + bbox.width / 2., tl[1] + bbox.height / 2.]
            head_centers2D[k] = center2D
            head_cameras[k] = cams[k]
    rays_o, rays_d = [], []
    for k in head_centers2D.keys():
        center2D = head_centers2D[k]
        camera = head_cameras[k]
        fx, fy, cx, cy = camera.inmat[0, 0], camera.inmat[1, 1], camera.inmat[0, 2], camera.inmat[1, 2]
        u, v = center2D
        direction = [(u - cx + 0.5) / fx, (v - cy + 0.5) / fy, 1.]
        ray_o = camera.c2w.tfmat[:3, 3]
        rays_o.append(ray_o)
        ray_d = direction @ camera.c2w.tfmat[:3, :3].T
        ray_d = ray_d / np.linalg.norm(ray_d)
        rays_d.append(ray_d)
    rays_o, rays_d = np.array(rays_o), np.array(rays_d)
    center3D = calculate_nearest_point_of_rays(rays_o[..., None], rays_d[..., None])
    if vis_path is not None:
        MRaysMesh(rays_o, rays_d, center3D).export(vis_path)
    head_centers2D = {}
    for k, v in cams.items():
        intrinsic = np.zeros([3, 4], dtype=np.float32)
        intrinsic[:, :3] = v.inmat
        extrinsic = v.c2w.tfmat
        extrinsic = np.linalg.inv(extrinsic)  # convert c2w to w2c
        points3d = np.zeros([1, 4, 1], dtype=np.float32)
        points3d[:, :3, 0] = center3D
        points3d[:, 3, 0] = 1.
        points2d = intrinsic @ extrinsic @ points3d
        pixels = points2d[:, :, 0]
        pixels = pixels / pixels[:, 2, None]
        pixels = pixels[:, :2]
        head_centers2D[k] = pixels[0]  # Only one pointpixels[0]  # Only one point
    return head_centers2D


def calculate_head_side_length(frames_dir, fbboxes, model_file, scale=1.2):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fp_model = load_BiSeNet_model(model_file, device=device)
    fp_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    fp_lut = np.zeros((256, ), dtype=np.uint8)  # 构建查找表, 默认替换值为0
    fp_lut[1:14] = 1  # 将标签对应的1-13的值(face区域)都设置为1
    fp_lut[17] = 1  # 将标签对应的17的值(hair区域)都设置为1
    head_hs = []
    head_ws = []
    hair_hs = []
    for k, v in fbboxes.items():
        if v is not None:
            frame_data = cv2.imread(osp.join(frames_dir, k))
            frame_data_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            tl = v.top_left
            tl_ = [tl[0] - v.width / 2., tl[1] - v.height / 2.]
            width_ = v.width * 2.
            height_ = v.height * 2.
            bbox = MFaceBBox(top_left=tl_, width=width_, height=height_, score=v.score, landmarks=v.landmarks)

            crp_frame_data = crop_frame(frame_data_rgb, bbox, bg_color=(255, 255, 255))
            crp_frame_gray = cv2.cvtColor(crp_frame_data, cv2.COLOR_RGB2GRAY)
            _, crp_frame_bin = cv2.threshold(crp_frame_gray, 0, 255, cv2.THRESH_OTSU)
            crp_frame_bin = 255 - crp_frame_bin
            contours, hierarchy = cv2.findContours(crp_frame_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            areas = []
            for i in range(len(contours)):
                areas.append(cv2.contourArea(contours[i]))
            max_idx = np.argmax(areas)
            max_contour = contours[max_idx]
            tlx, tly, _, _ = cv2.boundingRect(np.array(max_contour))
            hair_height = (bbox.height / 2. - tly) * 2.
            hair_hs.append(hair_height)

            crp_frame_ts = fp_transform(crp_frame_data).unsqueeze(0).to(device)
            with torch.no_grad():
                crp_frame_out = fp_model(crp_frame_ts)[0]
            crp_frame_out = crp_frame_out.squeeze(0).cpu().numpy().argmax(0)
            crp_frame_mask = cv2.LUT(crp_frame_out.astype(np.uint8), fp_lut)
            crp_frame_mask = crp_frame_mask * 255
            contours, hierarchy = cv2.findContours(crp_frame_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            areas = []
            for i in range(len(contours)):
                areas.append(cv2.contourArea(contours[i]))
            max_idx = np.argmax(areas)
            max_contour = contours[max_idx]
            _, _, w, h = cv2.boundingRect(np.array(max_contour))
            head_hs.append(h)
            head_ws.append(w)
    max_head_hs = max(head_hs)
    max_head_ws = max(head_ws)
    max_hair_hs = max(hair_hs)
    side_length = max(max(max_head_hs, max_head_ws), max_hair_hs) * scale
    return side_length


def visualize_head_bboxes(vfrm: MVideoFrames,
                          head_bboxes: Dict[str, MHeadBBox],
                          save_dir: str,
                          thickness: int = 2,
                          circle_radius: int = 2):
    os.makedirs(save_dir, exist_ok=True)
    bbox_color = (0, 255, 0)  # BGR
    point_color = (0, 0, 255)  # BGR
    for frame in tqdm.tqdm(vfrm.frames, desc=f'Head Detection Visualization for {vfrm.path}'):
        frame_path = osp.join(vfrm.path, frame)
        frame_data = cv2.imread(frame_path)
        bbox = head_bboxes[frame]
        assert bbox is not None, f'No head bbox for frame {frame}'
        tl = bbox.top_left
        br = [tl[0] + bbox.width, tl[1] + bbox.height]
        tl = np.int32(tl).tolist()
        br = np.int32(br).tolist()
        cv2.rectangle(frame_data, tl, br, bbox_color, thickness)
        center2D = [tl[0] + bbox.width / 2., tl[1] + bbox.height / 2.]
        center2D = np.int32(center2D).tolist()
        cv2.circle(frame_data, center2D, circle_radius, point_color, thickness)
        cv2.imwrite(osp.join(save_dir, frame), frame_data)
    return True


def align_head(frame_dir, frame_files, head_bboxes, save_dir, bg_color=(0, 0, 0)):
    """Move face center to center of image.

    Args:
        frame_dir (_type_): _description_
        frame_files (_type_): _description_
        face_centers (_type_): _description_
        save_dir (_type_): _description_
        bg_color (tuple, optional): _description_. Defaults to (0, 0, 0).

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    os.makedirs(save_dir, exist_ok=True)
    head_bboxes_ = {}
    for frame_file in tqdm.tqdm(frame_files, desc=f'Face Center Align for {frame_dir}'):
        frame_path = osp.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        tl = head_bboxes[frame_file].top_left
        width = head_bboxes[frame_file].width
        height = head_bboxes[frame_file].height
        center = [tl[0] + width / 2., tl[1] + height / 2.]
        h, w, _ = frame.shape
        frame_center = [w / 2., h / 2.]
        center = np.round(center).astype(np.int32).tolist()
        delta_xy = [frame_center[0] - center[0], frame_center[1] - center[1]]
        m = np.float64([[1, 0, delta_xy[0]], [0, 1, delta_xy[1]]])
        frame_ = cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
        cv2.imwrite(osp.join(save_dir, frame_file), frame_)
        tl_ = [frame_center[0] - width / 2., frame_center[1] - height / 2.]
        head_bboxes_[frame_file] = MHeadBBox(top_left=tl_, width=width, height=height)
    return head_bboxes_


def align_head_with_center(frame_dir, frame_files, center_dict, save_dir, bg_color=(0, 0, 0)):
    os.makedirs(save_dir, exist_ok=True)
    align_center_dict = {}
    for frame_file in tqdm.tqdm(frame_files, desc=f'Face Center Align for {frame_dir}'):
        frame_path = osp.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if bg_color is None:
            cur_bg_color = frame[0, 0, :]
            cur_bg_color = cur_bg_color.tolist()
        else:
            cur_bg_color = bg_color
        h, w, _ = frame.shape
        frame_center = [w / 2., h / 2.]
        # center = np.round(center_dict[frame_file]).astype(np.int32).tolist()
        center = center_dict[frame_file].tolist()
        delta_xy = [frame_center[0] - center[0], frame_center[1] - center[1]]
        m = np.float64([[1, 0, delta_xy[0]], [0, 1, delta_xy[1]]])
        frame_ = cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=cur_bg_color)
        cv2.imwrite(osp.join(save_dir, frame_file), frame_)
        align_center_dict[frame_file] = frame_center
    return align_center_dict


def detect_face_keypoints(vfrm: MVideoFrames, score_thres: float):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=device)
    frame_kps = {}
    frame_fps = {}
    for frame in vfrm.frames:
        frame_file = osp.join(vfrm.path, frame)
        print(frame_file, end=': ')
        landmarks = fa.get_landmarks_from_image(frame_file, return_landmark_score=True)
        if landmarks[0] is None:
            frame_kps[frame] = None
            frame_fps[frame] = None
            print('None')
        elif len(landmarks[0]) != 1:
            frame_kps[frame] = None
            frame_fps[frame] = None
            print('len(landmarks[0]) != 1')
        else:
            mean_score = landmarks[1][0].mean()
            if mean_score < score_thres:
                frame_kps[frame] = None
                frame_fps[frame] = None
                print(f'mean_score({mean_score}) < score_thres({score_thres})')
            else:
                frame_kps[frame] = MFaceKeyPoints(points=landmarks[0][0][:, :2], scores=landmarks[1][0])
                # IGNORE BEGIN
                eyebrow1_pt = landmarks[0][0][17:22].mean(axis=0)
                eyebrow1_sc = landmarks[1][0][17:22].mean()
                eyebrow2_pt = landmarks[0][0][22:27].mean(axis=0)
                eyebrow2_sc = landmarks[1][0][22:27].mean()
                nose_pt = landmarks[0][0][27:31].mean(axis=0)
                nose_sc = landmarks[1][0][27:31].mean()
                nostril_pt = landmarks[0][0][31:36].mean(axis=0)
                nostril_sc = landmarks[1][0][31:36].mean()
                eye1_pt = landmarks[0][0][36:42].mean(axis=0)
                eye1_sc = landmarks[1][0][36:42].mean()
                eye2_pt = landmarks[0][0][42:48].mean(axis=0)
                eye2_sc = landmarks[1][0][42:48].mean()
                mouth_pt = landmarks[0][0][48:68].mean(axis=0)
                mouth_sc = landmarks[1][0][48:68].mean()
                feat_pts = np.array([eyebrow1_pt, eyebrow2_pt, nose_pt, nostril_pt, eye1_pt, eye2_pt, mouth_pt])
                feat_scs = np.array([eyebrow1_sc, eyebrow2_sc, nose_sc, nostril_sc, eye1_sc, eye2_sc, mouth_sc])
                # IGNORE END
                frame_fps[frame] = MFaceKeyPoints(points=feat_pts[:, :2], scores=feat_scs)
                print(mean_score)
    return frame_kps, frame_fps


def detect_fv_face_keypoints(vfrm: MVideoFrames, fv_list: list, score_thres: float):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    frame_kps = {}
    frame_fps = {}
    for frame in vfrm.frames:
        if frame not in fv_list:
            frame_kps[frame] = None
            frame_fps[frame] = None
            continue
        frame_file = osp.join(vfrm.path, frame)
        print(frame_file, end=': ')
        landmarks = fa.get_landmarks_from_image(frame_file, return_landmark_score=True)
        if landmarks[0] is None:
            frame_kps[frame] = None
            frame_fps[frame] = None
            print('None')
        elif len(landmarks[0]) != 1:
            frame_kps[frame] = None
            frame_fps[frame] = None
            print('len(landmarks[0]) != 1')
        else:
            mean_score = landmarks[1][0].mean()
            if mean_score < score_thres:
                frame_kps[frame] = None
                frame_fps[frame] = None
                print(f'mean_score({mean_score}) < score_thres({score_thres})')
            else:
                frame_kps[frame] = MFaceKeyPoints(points=landmarks[0][0][:, :2], scores=landmarks[1][0])
                eyebrow1_pt = landmarks[0][0][17:22].mean(axis=0)
                eyebrow1_sc = landmarks[1][0][17:22].mean()
                eyebrow2_pt = landmarks[0][0][22:27].mean(axis=0)
                eyebrow2_sc = landmarks[1][0][22:27].mean()
                nose_pt = landmarks[0][0][27:31].mean(axis=0)
                nose_sc = landmarks[1][0][27:31].mean()
                nostril_pt = landmarks[0][0][31:36].mean(axis=0)
                nostril_sc = landmarks[1][0][31:36].mean()
                eye1_pt = landmarks[0][0][36:42].mean(axis=0)
                eye1_sc = landmarks[1][0][36:42].mean()
                eye2_pt = landmarks[0][0][42:48].mean(axis=0)
                eye2_sc = landmarks[1][0][42:48].mean()
                mouth_pt = landmarks[0][0][48:68].mean(axis=0)
                mouth_sc = landmarks[1][0][48:68].mean()
                feat_pts = np.array([eyebrow1_pt, eyebrow2_pt, nose_pt, nostril_pt, eye1_pt, eye2_pt, mouth_pt])
                feat_scs = np.array([eyebrow1_sc, eyebrow2_sc, nose_sc, nostril_sc, eye1_sc, eye2_sc, mouth_sc])
                frame_fps[frame] = MFaceKeyPoints(points=feat_pts[:, :2], scores=feat_scs)
                print(mean_score)
    return frame_kps, frame_fps


def visualize_face_keypoints(vfrm: MVideoFrames,
                             frame_kps: Dict[str, MFaceKeyPoints],
                             save_dir: str,
                             thickness: int = 2,
                             circle_radius: int = 2):
    os.makedirs(save_dir, exist_ok=True)
    pred_type = namedtuple('prediction_type', ['slice', 'color'])
    pred_types = {
        'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
        'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
        'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
        'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
        'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
        'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
        'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
        'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
        'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
    }
    for frame in vfrm.frames:
        frame_file = osp.join(vfrm.path, frame)
        frame_data = cv2.imread(frame_file)
        frame_kp = frame_kps[frame]
        if frame_kp is not None:
            frame_kp_points = frame_kps[frame].points.astype(np.int32)
            for pred_type in pred_types.values():
                color = (np.array(pred_type.color[:3]) * 255).astype(np.int32).tolist()
                for pt in frame_kp_points[pred_type.slice]:
                    cv2.circle(frame_data, pt, circle_radius, color, thickness)
        cv2.imwrite(osp.join(save_dir, frame), frame_data)
    return True


def visualize_face_featpoints(vfrm: MVideoFrames,
                              frame_kps: Dict[str, MFaceKeyPoints],
                              save_dir: str,
                              thickness: int = 2,
                              circle_radius: int = 2):
    os.makedirs(save_dir, exist_ok=True)
    pred_type = namedtuple('prediction_type', ['slice', 'color'])
    pred_types = [(1.0, 0.498, 0.055, 0.4), (1.0, 0.498, 0.055, 0.4), (0.345, 0.239, 0.443, 0.4),
                  (0.345, 0.239, 0.443, 0.4), (0.596, 0.875, 0.541, 0.3), (0.596, 0.875, 0.541, 0.3),
                  (0.596, 0.875, 0.541, 0.3)]
    for frame in vfrm.frames:
        frame_file = osp.join(vfrm.path, frame)
        frame_data = cv2.imread(frame_file)
        frame_kp = frame_kps[frame]
        if frame_kp is not None:
            frame_kp_points = frame_kps[frame].feat_pts.astype(np.int32)
            for i, pred_type in enumerate(pred_types):
                color = (np.array(pred_type[:3]) * 255).astype(np.int32).tolist()
                cv2.circle(frame_data, frame_kp_points[i], circle_radius, color, thickness)
        cv2.imwrite(osp.join(save_dir, frame), frame_data)
    return True


def calculate_face_points_3d(kps2D, cams, frame_type, vis_dir=None):
    assert frame_type in ['colmap', 'nl3dmm'], f'frame_type must be colmap or nl3dmm, but got {frame_type}'
    pt_num = None
    for k, v in kps2D.items():
        if v is not None:
            pt_num = v.points.shape[0]
            break
    centers3D = []
    for pt_id in range(pt_num):
        centers2D = {}
        cameras = {}
        for k in kps2D.keys():
            if kps2D[k] is not None:
                centers2D[k] = kps2D[k].points[pt_id]
                cameras[k] = cams[frame_type][k]
        rays_o, rays_d = [], []
        for k in centers2D.keys():
            center2D = centers2D[k]
            camera = cameras[k]
            fx, fy, cx, cy = camera.inmat[0, 0], camera.inmat[1, 1], camera.inmat[0, 2], camera.inmat[1, 2]
            u, v = center2D
            direction = [(u - cx + 0.5) / fx, (v - cy + 0.5) / fy, 1.]
            ray_o = camera.c2w.tfmat[:3, 3]
            rays_o.append(ray_o)
            ray_d = direction @ camera.c2w.tfmat[:3, :3].T
            ray_d = ray_d / np.linalg.norm(ray_d)
            rays_d.append(ray_d)
        rays_o, rays_d = np.array(rays_o), np.array(rays_d)
        center3D = calculate_nearest_point_of_rays(rays_o[..., None], rays_d[..., None])
        if vis_dir is not None:
            os.makedirs(vis_dir, exist_ok=True)
            MRaysMesh(rays_o, rays_d, center3D).export(osp.join(vis_dir, f'{frame_type}_{pt_id:02d}.ply'))
        centers3D.append(center3D)
    centers3D = np.array(centers3D)
    print(f'{frame_type} centers3D.shape: {centers3D.shape}')
    return centers3D


def simil_analysis(X0, X1):
    # m, r, t = msimil.process(source_points, target_points)
    m, r, t = msimil.process(X1, X0)
    print(f'm:\n{m}\nr: {r}\nt:\n{t}')
    T1to0 = np.eye(4)
    T1to0[:3, :3] = m * r
    T1to0[:3, 3:] = t
    print(f'T1to0:\n{T1to0}')
    return T1to0


def transform_points(pts, tfmatrix):
    if pts.shape[1] == 3:
        pts_ = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    elif pts.shape[1] == 4:
        pts_ = pts
    else:
        raise ValueError(f'pts.shape[1] must be 3 or 4, but got {pts.shape[1]}')
    tfpts = (tfmatrix @ pts_.T).T[:, :3]
    return tfpts


def transform_cam(cam, tfmatrix):
    T_c2w = cam.c2w.tfmat
    T_c2w_ = tfmatrix @ T_c2w
    T_w2c_ = np.linalg.inv(T_c2w_)
    c2w_ = MTransMatrix(rotmat=T_c2w_[:3, :3], tvec=T_c2w_[:3, 3], tfmat=T_c2w_)
    w2c_ = MTransMatrix(rotmat=T_w2c_[:3, :3], tvec=T_w2c_[:3, 3], tfmat=T_w2c_)
    cam_ = MCamera(model=cam.model, width=cam.width, height=cam.height, inmat=cam.inmat, c2w=c2w_, w2c=w2c_, bd=cam.bd)
    return cam_


def colorize_frame_label(label):
    def bgr_color(v):
        return (np.array(plt.cm.tab20(v)[:3]) * 255).astype(np.uint8)[::-1]

    colorized = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for c in np.unique(label):
        colorized[label == c] = bgr_color(c)
    return colorized


def filter_small_regions(mask, threshold=0.1, kernel_size=(3, 3), close_iterations=5):
    if np.sum(mask) == 0:
        return copy.deepcopy(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    closed = cv2.morphologyEx(copy.deepcopy(mask), cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    label = measure.label(closed, connectivity=2)
    props = measure.regionprops(label)
    area_pixels = [p.area for p in props]
    area_threshold = np.mean(area_pixels) * threshold
    mask_ = np.zeros_like(mask)
    for p in props:
        if p.area >= area_threshold:
            p_mask = (label == p.label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(p_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
            p_mask_ = cv2.drawContours(np.zeros_like(p_mask), contours, 0, 255, cv2.FILLED)
            mask_[p_mask_ > 0] = 255
    return mask_


def parse_face(wvfrm, mvfrm, fp_model_path, class2label, threshold=0.1, kernel_size=(3, 3), close_iterations=5):
    fp_model = load_BiSeNet_model(fp_model_path, device=device)
    fp_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    results = {}
    for frame in tqdm.tqdm(wvfrm.frames, desc=f'Face Parsing for {wvfrm.path}'):
        frame_data_path = osp.join(wvfrm.path, frame)
        frame_data = Image.open(frame_data_path)
        ori_wh = (frame_data.width, frame_data.height)
        frame_data = frame_data.resize((512, 512), Image.BILINEAR)
        frame_data = np.array(frame_data)

        frame_mask_path = osp.join(mvfrm.path, frame)
        frame_mask = cv2.imread(frame_mask_path)

        frame_ts = fp_transform(frame_data).unsqueeze(0).to(device)
        with torch.no_grad():
            frame_pred = fp_model(frame_ts)[0]
        frame_pred_label = frame_pred.squeeze(0).cpu().numpy().argmax(0)
        frame_label = np.zeros_like(frame_pred_label)
        for src_num, src_label in bisenet_num2label.items():
            dst_num = ours_label2num[src_label]
            cur_mask = (frame_pred_label == src_num).astype(np.uint8) * 255
            cur_mask_ = filter_small_regions(cur_mask)
            frame_label[cur_mask_ != 0] = dst_num

        frame_label = Image.fromarray(frame_label)
        frame_label = frame_label.resize(ori_wh, Image.NEAREST)
        frame_label = np.array(frame_label)

        frame_mask_gray = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
        _, frame_mask_bin = cv2.threshold(frame_mask_gray, 0, 255, cv2.THRESH_OTSU)
        frame_label[frame_mask_bin == 0] = ours_label2num['background']

        results[frame] = frame_label

    return results


def calc_trans_bbox_side(cur_dst_lmx4_5p, cur_src_lmx4_5p, crop_wvfrmx4):
    d2sx4_tf = SimilarityTransform()
    d2sx4_tf.estimate(cur_dst_lmx4_5p, cur_src_lmx4_5p)

    d2sx4_tf_bbox = d2sx4_tf(
        np.array([[0, 0], [crop_wvfrmx4.shape[0], 0], [crop_wvfrmx4.shape[0], crop_wvfrmx4.shape[1]],
                  [0, crop_wvfrmx4.shape[1]]]))
    d2sx4_tf_bbox_min = np.min(d2sx4_tf_bbox, axis=0)
    d2sx4_tf_bbox_max = np.max(d2sx4_tf_bbox, axis=0)
    d2sx4_tf_bbox_w = d2sx4_tf_bbox_max[0] - d2sx4_tf_bbox_min[0]
    d2sx4_tf_bbox_h = d2sx4_tf_bbox_max[1] - d2sx4_tf_bbox_min[1]
    d2sx4_tf_bbox_w = np.round(np.max([d2sx4_tf_bbox_w, d2sx4_tf_bbox_h]))
    return d2sx4_tf_bbox_w


def calc_head_bbox_side(mask, scale, cropx4_bbox, center_xy):
    m_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(m_gray, 0, 255, cv2.THRESH_OTSU)

    fp_bbox_tlx, fp_bbox_tly, fp_bbox_w, fp_bbox_h = cv2.boundingRect(mask_bin)
    fp_bbox_tly = fp_bbox_tly + cropx4_bbox.top_left[1]  # 只考虑头顶
    fp_bbox_height = abs(center_xy[1] - fp_bbox_tly) * 2
    fp_bbox_height = np.round(fp_bbox_height * scale)
    return fp_bbox_height


def refine_frame_label(frame_label, frame_mask, class2label, threshold=0.1, kernel_size=(3, 3), close_iterations=5):
    ebro_mask = (frame_label == class2label['eyebrow']).astype(np.uint8) * 255
    ebro_mask_ = filter_small_regions(ebro_mask, threshold, kernel_size, close_iterations)

    eyes_mask = (frame_label == class2label['eye']).astype(np.uint8) * 255
    eyes_mask_ = filter_small_regions(eyes_mask, threshold, kernel_size, close_iterations)

    nose_mask = (frame_label == class2label['nose']).astype(np.uint8) * 255
    nose_mask_ = filter_small_regions(nose_mask, threshold, kernel_size, close_iterations)

    mout_mask = (frame_label == class2label['mouth']).astype(np.uint8) * 255
    mout_mask_ = filter_small_regions(mout_mask, threshold, kernel_size, close_iterations)

    lips_mask = (frame_label == class2label['lip']).astype(np.uint8) * 255
    lips_mask_ = filter_small_regions(lips_mask, threshold, kernel_size, close_iterations)

    ears_mask = (frame_label == class2label['ear']).astype(np.uint8) * 255
    ears_mask_ = filter_small_regions(ears_mask, threshold, kernel_size, close_iterations)

    face_mask = np.ones_like(frame_label, dtype=bool)
    face_mask *= frame_label != class2label['background']
    face_mask *= frame_label != class2label['uncertainty']
    face_mask *= frame_label != class2label['hair']
    face_mask = face_mask.astype(np.uint8) * 255
    face_mask_ = filter_small_regions(face_mask, threshold, kernel_size, close_iterations)

    skin_mask_ = copy.deepcopy(face_mask_)
    skin_mask_[ebro_mask_ > 0] = 0
    skin_mask_[eyes_mask_ > 0] = 0
    skin_mask_[nose_mask_ > 0] = 0
    skin_mask_[mout_mask_ > 0] = 0
    skin_mask_[lips_mask_ > 0] = 0
    skin_mask_[ears_mask_ > 0] = 0

    hair_mask = (frame_label == class2label['hair']).astype(np.uint8) * 255
    hair_mask_ = filter_small_regions(hair_mask, threshold, kernel_size, close_iterations)
    hair_mask_[face_mask_ > 0] = 0

    frame_mask_gray = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
    _, frame_mask_bin = cv2.threshold(frame_mask_gray, 0, 255, cv2.THRESH_OTSU)

    background_mask_ = np.ones_like(frame_label, dtype=bool)
    background_mask_ *= frame_label == class2label['background']
    background_mask_ *= frame_mask_bin == 0
    background_mask_ = background_mask_.astype(np.uint8) * 255

    uncertainty_mask_ = np.ones_like(frame_label, dtype=bool)
    uncertainty_mask_[background_mask_ > 0] = False
    uncertainty_mask_[skin_mask_ > 0] = False
    uncertainty_mask_[ebro_mask_ > 0] = False
    uncertainty_mask_[eyes_mask_ > 0] = False
    uncertainty_mask_[nose_mask_ > 0] = False
    uncertainty_mask_[mout_mask_ > 0] = False
    uncertainty_mask_[lips_mask_ > 0] = False
    uncertainty_mask_[ears_mask_ > 0] = False
    uncertainty_mask_[hair_mask_ > 0] = False
    uncertainty_mask_ = uncertainty_mask_.astype(np.uint8) * 255

    frame_label_ = np.zeros_like(frame_label)
    frame_label_[background_mask_ > 0] = class2label['background']
    frame_label_[skin_mask_ > 0] = class2label['skin']
    frame_label_[ebro_mask_ > 0] = class2label['eyebrow']
    frame_label_[eyes_mask_ > 0] = class2label['eye']
    frame_label_[nose_mask_ > 0] = class2label['nose']
    frame_label_[mout_mask_ > 0] = class2label['mouth']
    frame_label_[lips_mask_ > 0] = class2label['lip']
    frame_label_[ears_mask_ > 0] = class2label['ear']
    frame_label_[hair_mask_ > 0] = class2label['hair']
    frame_label_[uncertainty_mask_ > 0] = class2label['uncertainty']

    return frame_label_


def parse_ours_face(wvfrm, mvfrm, fp_conf, class2label, threshold=0.1, kernel_size=(3, 3), close_iterations=5):
    assert fp_conf['type'] in ['BiSeNet', 'ibug'], f'Unknown face parsing type: {fp_conf["type"]}'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if fp_conf['type'] == 'BiSeNet':
        fp_model = load_BiSeNet_model(fp_conf['ckpt'], device=device)
        fp_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        fp_lut = np.zeros((256, ), dtype=np.uint8)
        fp_lut[:19] = [0, 1, 2, 2, 3, 3, 1, 7, 7, 7, 4, 5, 6, 6, 0, 0, 0, 8, 8]
    elif fp_conf['type'] == 'ibug':
        fp_model = FaceParser(device=device,
                              ckpt=fp_conf['ckpt'],
                              encoder=fp_conf['encoder'],
                              decoder=fp_conf['decoder'],
                              num_classes=fp_conf['num_classes'])
        if fp_conf['num_classes'] == 14:
            fp_lut = np.zeros((256, ), dtype=np.uint8)
            fp_lut[:14] = [0, 1, 2, 2, 3, 3, 4, 6, 5, 6, 8, 7, 7, 1]
        else:
            raise RuntimeError(f'Unknown face parsing num_classes: {fp_conf["num_classes"]}')
    else:
        raise RuntimeError(f'Unknown face parsing type: {fp_conf["type"]}')

    results = {}
    for frame in tqdm.tqdm(wvfrm.frames, desc=f'Face Parsing for {wvfrm.path}'):
        frame_data_path = osp.join(wvfrm.path, frame)
        frame_data = Image.open(frame_data_path)
        ori_wh = (frame_data.width, frame_data.height)
        frame_data = frame_data.resize((512, 512), Image.BILINEAR)
        frame_data = np.array(frame_data)

        frame_mask_path = osp.join(mvfrm.path, frame)
        frame_mask = cv2.imread(frame_mask_path)
        frame_mask = cv2.resize(frame_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        if fp_conf['type'] == 'BiSeNet':
            frame_ts = fp_transform(frame_data).unsqueeze(0).to(device)
            with torch.no_grad():
                frame_pred = fp_model(frame_ts)[0]
            frame_pred_label = frame_pred.squeeze(0).cpu().numpy().argmax(0)
            frame_label = cv2.LUT(frame_pred_label.astype(np.uint8), fp_lut)
        elif fp_conf['type'] == 'ibug':
            h, w = frame_data.shape[:2]
            frame_bboxes = np.array([[0, 0, w - 1, h - 1]])
            frame_pred = fp_model.predict_img(frame_data, frame_bboxes, rgb=True)
            frame_pred_label = frame_pred[0]
            frame_label = cv2.LUT(frame_pred_label.astype(np.uint8), fp_lut)
        else:
            raise RuntimeError(f'Unknown face parsing type: {fp_conf["type"]}')

        frame_label_ = refine_frame_label(frame_label, frame_mask, class2label, threshold, kernel_size,
                                          close_iterations)
        frame_label_ = Image.fromarray(frame_label_)
        frame_label_ = frame_label_.resize(ori_wh, Image.NEAREST)
        frame_label_ = np.array(frame_label_)
        results[frame] = frame_label_
    return results


def aggregate_fp_labels(label_list, mvfrm, class2label, threshold=0.1, kernel_size=(3, 3), close_iterations=5):
    results = {}
    for frame in tqdm.tqdm(mvfrm.frames, desc=f'Label Aggregating for {mvfrm.path}'):
        frame_mask_path = osp.join(mvfrm.path, frame)
        frame_mask = cv2.imread(frame_mask_path)

        frame_label_list = []
        for i in range(len(label_list)):
            frame_label_list.append(label_list[i][frame][:, :, None])
        frame_labels = np.concatenate(frame_label_list, axis=2)
        frame_label_count = np.zeros(list(frame_labels.shape[:2]) + [frame_labels.max() + 1], dtype=np.uint8)
        for i in range(frame_labels.max() + 1):
            frame_label_count[:, :, i] = np.sum(frame_labels == i, axis=2)
        frame_label = np.argmax(frame_label_count, axis=2).astype(np.uint8)
        frame_label_ = refine_frame_label(frame_label, frame_mask, class2label, threshold, kernel_size,
                                          close_iterations)
        results[frame] = frame_label_
    return results
