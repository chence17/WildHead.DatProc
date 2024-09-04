import cv2
import numpy as np
import os.path as osp
import os
import tqdm
import json
from dpestimator import HeadPoseEstimator
from scipy.spatial.transform import Rotation


def eg3dcamparams(R_in):
    # World Coordinate System: x(right), y(up), z(forward)
    camera_dist = 2.7
    intrinsics = np.array([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]])
    # assume inputs are rotation matrices for world2cam projection
    R = np.array(R_in).astype(np.float32).reshape(4,4)
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
    label_new = np.concatenate([cam2world.reshape(16), intrinsics.reshape(9)], -1)
    return label_new


def hpose2R(hpose):
    yaw, roll, pitch = hpose
    r_pitch, r_yaw, r_roll = -pitch, -yaw, -roll
    R = Rotation.from_euler('zyx', [r_roll, r_yaw, r_pitch], degrees=True).as_matrix()
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


hed_pe = HeadPoseEstimator(weights_file='assets/whenet_1x3x224x224_prepost.onnx')

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
            continue

        inpaint_poses = {}
        for inpaint_file in tqdm.tqdm(inpaint_files):
            try:
                head_image = cv2.imread(osp.join(inpaint_dir, inpaint_file))
                hpose = hed_pe(head_image, isBGR=True)
                camera_poses = hpose2camera(hpose)

                hpose_list = hpose.astype(np.float32).tolist()
                camera_poses_list = camera_poses.tolist()
            except Exception as e:
                print(f"Error: {inpaint_file}")
                print(e)
                hpose_list = []
                camera_poses_list = []
            inpaint_poses[inpaint_file] = {"hpose": hpose_list, "camera": camera_poses_list}
        with open(inpaint_poses_json, "w") as f:
            json.dump(inpaint_poses, f)
        print(f"Saved: {inpaint_poses_json}")
