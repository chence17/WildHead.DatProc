import cv2
import numpy as np
import os.path as osp
import os
import tqdm
import json
from dpdetector import FaceAlignmentDetector
from dpestimator import HeadPoseEstimator
from dpcropper import FrontViewCropperV2
from scipy.spatial.transform import Rotation
from skimage import io


def eg3dcamparams(R_in):
    # World Coordinate System: x(right), y(up), z(forward)
    camera_dist = 2.7
    intrinsics = np.array([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]])
    # assume inputs are rotation matrices for world2cam projection
    R = np.array(R_in).astype(np.float32).reshape(4, 4)
    # add camera translation
    t = np.eye(4, dtype=np.float32)
    t[2, 3] = - camera_dist

    # convert to OpenCV camera
    convert = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]).astype(np.float32)

    # world2cam -> cam2world
    P = convert @ t @ R
    cam2world = np.linalg.inv(P)

    # add intrinsics
    label_new = np.concatenate(
        [cam2world.reshape(16), intrinsics.reshape(9)], -1)
    return label_new


def hpose2R(hpose):
    yaw, roll, pitch = hpose
    r_pitch, r_yaw, r_roll = -pitch, -yaw, -roll
    R = Rotation.from_euler(
        'zyx', [r_roll, r_yaw, r_pitch], degrees=True).as_matrix()
    return R


def hpose2camera(hpose):
    R = hpose2R(hpose)
    s = 1
    t3d = np.array([0., 0., 0.])
    R[:, :3] = R[:, :3] * s
    P = np.concatenate([R, t3d[:, None]], 1)
    P = np.concatenate([P, np.array([[0, 0, 0, 1.]])], 0)
    cam = eg3dcamparams(P.flatten())
    return cam


def R2hpose(R_matrix):
    """degree"""
    r_hpose = Rotation.from_matrix(R_matrix).as_euler('zyx', degrees=True)
    r_roll, r_yaw, r_pitch = r_hpose.astype(np.float32).tolist()
    pitch, yaw, roll = -r_pitch, -r_yaw, -r_roll
    hpose = [yaw, roll, pitch]
    return hpose


# inverse convert from OpenCV camera
convert = np.array([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]]).astype(np.float32)
inv_convert = np.linalg.inv(convert)

flmk_det = FaceAlignmentDetector(score_thres=0.8)
fv_crpr = FrontViewCropperV2(
    config_file='TDDFA_V2/configs/mb1_120x120.yml', mode='gpu')
hed_pe = HeadPoseEstimator(
    weights_file='assets/whenet_1x3x224x224_prepost.onnx')

img_root_dir = "/data/K-Hairstyle/debug/rawset_crop/0003.rawset"
pos1_list = sorted(os.listdir(img_root_dir))
for pos1 in pos1_list:
    print(f"==>> pos1: {pos1}")
    pos2_list = sorted(os.listdir(osp.join(img_root_dir, pos1)))
    for pos2 in pos2_list:
        print(f"==>> pos2: {pos2}")
        img_dir = osp.join(img_root_dir, pos1, pos2)
        inpaint_dir = osp.join(img_dir, "image1024_inpaint")
        if not osp.exists(inpaint_dir):
            print(f"[Error] Directory not found: {inpaint_dir}")
            continue
        inpaint_files = sorted(os.listdir(inpaint_dir))
        inpaint_poses_json = osp.join(img_dir, "image1024_inpaint_poses.json")
        if osp.exists(inpaint_poses_json):
            print(f"[Warning] File exists: {inpaint_poses_json}")
            continue

        inpaint_poses = {}
        for inpaint_file in tqdm.tqdm(inpaint_files):
            try:
                # head_image = cv2.imread(osp.join(inpaint_dir, inpaint_file))
                # RGB uint8 HW3 ndarray
                head_image = io.imread(osp.join(inpaint_dir, inpaint_file))
                landmarks = flmk_det(head_image, isBGR=False)
                if landmarks is None:
                    print(f"Lanmarks not found: {inpaint_file}")
                    hpose = hed_pe(head_image, isBGR=False)
                    camera_poses = hpose2camera(hpose)

                    hpose_list = hpose.astype(np.float32).tolist()
                    camera_poses_list = camera_poses.tolist()
                else:
                    print(f"Lanmarks found: {inpaint_file}")
                    cropped_img, camera_poses, quad, tf_quad, none_padding_ratio, cropped_pad_mask = fv_crpr(
                        head_image.copy(), landmarks)
                    quad, tf_quad = np.array(quad), np.array(tf_quad)

                    # cam2world
                    cam2world = camera_poses[:16].reshape(4, 4)
                    P = np.linalg.inv(cam2world)
                    R_in = inv_convert @ P
                    R_in = R_in[:3, :3]
                    hpose = R2hpose(R_in)

                    hpose_list = hpose
                    camera_poses_list = camera_poses.tolist()
            except Exception as e:
                print(f"Error: {inpaint_file}")
                print(e)
                hpose_list = []
                camera_poses_list = []
            inpaint_poses[inpaint_file] = {
                "hpose": hpose_list, "camera": camera_poses_list}
        with open(inpaint_poses_json, "w") as f:
            json.dump(inpaint_poses, f)
        print(f"Saved: {inpaint_poses_json}")
