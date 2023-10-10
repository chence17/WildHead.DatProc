'''
Author: tianhao 120090472@link.cuhk.edu.cn
Date: 2023-10-10 16:44:14
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-10-10 21:58:52
FilePath: /DatProc/lq_image_cleanup_scripts/remove_wrong_campose.py
Description: 
This script removes the wrong campose images from the dataset.json file.
For LPFF, FFHQ, no image should be back facing.
For K-Hairstyle, no image should be front facing.

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import os
import json
import argparse
import numpy as np
from tqdm import tqdm

def config_parser():
    parser = argparse.ArgumentParser(description='This script removes the wrong campose images from the dataset.json file.')
    parser.add_argument('--src_json', type=str, default='/data1/chence/PanoHeadData/single_view/dataset.json', help='The source json file.')
    parser.add_argument('--filter_view', type=str, default='back', choices=['back', 'front'], help='The view to be filtered.')
    return parser.parse_args()

def get_cam_coords(c2w):
    # copied from V1.data_source_khs.ipynb
    # World Coordinate System: x(right), y(up), z(forward)
    T = c2w[:3, 3]
    x, y, z = T
    r = np.sqrt(x**2+y**2+z**2)
    # theta = np.rad2deg(np.arctan(np.sqrt(x**2+z**2)/y))
    theta = np.rad2deg(np.arctan2(x, z))
    if theta >= -90 and theta <= 90:
        theta += 90
    elif theta>=-180 and theta < -90:
        theta += 90
    elif theta>90 and theta <= 180:
        theta -= 270
    else:
        raise ValueError('theta out of range')
    # phi = np.rad2deg(np.arctan(z/x))+180
    phi = np.rad2deg(np.arccos(y/r))
    return [theta, phi, r, x, y, z] # [:3] sperical cood, [3:] cartesian cood

args =  config_parser()
assert os.path.exists(args.src_json), f'{args.src_json} does not exist.'
save_json_path = args.src_json.replace('.json', f'_no_{args.filter_view}.json')
with open(args.src_json, 'r') as f:
    json_origin = json.load(f)
json_no_wrong_campose = {}
# process
has_more_head_counter = 0
print(f'Before processing, there are {len(json_origin)} images.')
for image, image_meta in tqdm(json_origin.items()):
    # make sure there is only one head left
    if len(image_meta["head"].keys()) > 1:
        print(f'{image} has more than one head.')
        has_more_head_counter += 1
        continue
    # load camera parameter
    for head_id, head_meta in image_meta["head"].items():
        cam_param = head_meta["camera"]
        # calculate spacial coord, get theta and phi
        c2w = np.array(cam_param[:16]).reshape(4, 4)
        theta, phi, r, x, y, z = get_cam_coords(c2w)
        # from theta and phi, get viewing direction
        if (
            (
                (theta >= 45 and theta <= 135) and # calculated front
                (args.filter_view == 'front')      # and want to filter front
            ) or (
                (theta >= -135 and theta <= -45) and # calculated back
                (args.filter_view == 'back')        # and want to filter back
            )
        ): # calculated front and filter front
            continue
        json_no_wrong_campose[image] = image_meta
print(f'Afer processing, there are {len(json_no_wrong_campose)} images.')
print(f'Ignoring {has_more_head_counter} images with more than one head.')
with open(save_json_path, 'w') as f:
    json.dump(json_no_wrong_campose, f, indent=2)
