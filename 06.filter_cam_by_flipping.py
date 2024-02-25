'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-11-11 15:37:40
LastEditors: tianhao 120090472@link.cuhk.edu.cn
LastEditTime: 2023-11-11 17:16:10
FilePath: /DatProc/06.filter_cam_by_flipping.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json, os, tqdm
import cv2
from dpestimator import HeadPoseEstimator

dataset_path = '/data2/chence/PanoHeadData/single_view_hq/dataset.json'
pe_info_path = '/data2/chence/PanoHeadData/single_view_hq/align_images/pose_estimation_info.json'
print(f'Loding dataset from {dataset_path}...', end='')
with open(dataset_path, 'r') as f:
    dataset_json = json.load(f)
image_names = list(dataset_json.keys())
print('Done')
pe = HeadPoseEstimator(weights_file = 'assets/whenet_1x3x224x224_prepost.onnx')
pose_estimation_info = {}

for image_name in tqdm.tqdm(image_names):
    image_pose_dict = {}
    image_name = image_name.replace('.png', '.jpg')
    image_path = os.path.join('/data2/chence/PanoHeadData/single_view_hq/align_images', image_name)
    image_data = cv2.imread(image_path)
    image_data_flipped = cv2.flip(image_data, 1)
    yaw, roll, pitch = pe(image_data, isBGR=True).tolist()
    yaw_bar, roll_bar, pitch_bar = pe(image_data_flipped, isBGR=True).tolist()
    image_pose_dict['yaw'] = yaw
    image_pose_dict['yaw_bar'] = yaw_bar
    image_pose_dict['pitch'] = pitch
    image_pose_dict['pitch_bar'] = pitch_bar
    image_pose_dict['roll'] = roll
    image_pose_dict['roll_bar'] = roll_bar
    pose_estimation_info[image_name] = image_pose_dict

with open(pe_info_path, 'w') as f:
    json.dump(pose_estimation_info, f, indent=4)