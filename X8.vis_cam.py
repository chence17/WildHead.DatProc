import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

meta = load_json('/data/PanoHeadData/single_view_hq/dataset.json')

length=2.7
eps=0.05

import numpy as np
from utils.visualize_utils import MMeshMeta

k = list(meta.keys())[0]
v = meta[k]
exmat = np.array(v['camera'][:16]).reshape(4, 4)
m = MMeshMeta()
m.create_camera_coordinates(exmat, length=length, eps=eps)
m.save('mvis_cam_single.ply')

import random

sample_num = 50
sampled_keys = random.sample(list(meta.keys()), sample_num)
m = MMeshMeta()
for k in sampled_keys:
    v = meta[k]
    if v['view'] == 'front':
        exmat = np.array(v['camera'][:16]).reshape(4, 4)
        m.create_camera_coordinates(exmat, length=length, eps=eps)
m.save(f'mvis_cam_multiple_{sample_num}_fv.ply')

# sample_num = 50
# sampled_keys = random.sample(list(meta.keys()), sample_num)
# m = MMeshMeta()
# for k in sampled_keys:
#     v = meta[k]
#     exmat = np.array(v['camera'][:16]).reshape(4, 4)
#     m.create_camera_coordinates(exmat, length=length, eps=eps)
# m.save(f'mvis_cam_multiple_{sample_num}.ply')

# sample_num = 100
# sampled_keys = random.sample(list(meta.keys()), sample_num)
# m = MMeshMeta()
# for k in sampled_keys:
#     v = meta[k]
#     exmat = np.array(v['camera'][:16]).reshape(4, 4)
#     m.create_camera_coordinates(exmat, length=length, eps=eps)
# m.save(f'mvis_cam_multiple_{sample_num}.ply')

# sample_num = 1000
# sampled_keys = random.sample(list(meta.keys()), sample_num)
# m = MMeshMeta()
# for k in sampled_keys:
#     v = meta[k]
#     exmat = np.array(v['camera'][:16]).reshape(4, 4)
#     m.create_camera_coordinates(exmat, length=length, eps=eps)
# m.save(f'mvis_cam_multiple_{sample_num}.ply')


# sample_num = 10000
# sampled_keys = random.sample(list(meta.keys()), sample_num)
# m = MMeshMeta()
# for k in sampled_keys:
#     v = meta[k]
#     exmat = np.array(v['camera'][:16]).reshape(4, 4)
#     m.create_camera_coordinates(exmat, length=length, eps=eps)
# m.save(f'mvis_cam_multiple_{sample_num}.ply')
