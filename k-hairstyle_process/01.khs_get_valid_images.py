'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-09-25 21:21:30
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-09-28 19:35:14
FilePath: /DatProc/k-hairstyle_process/01.khs_get_valid_images.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
"""
Since the front view images of k-hairstyle is masked, we can only use the back view images in our dataset
This script traverse khs dataset and creates a json specifying all the back view images, together with some of their meta info
"""
import os
import re
import cv2
import json
import imagesize
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def get_json_files_from(dir_path):
    json_file_paths = []
    for _file in os.listdir(dir_path):
        if _file.endswith('.json'):
            json_file_paths.append(os.path.join(dir_path, _file))
    return json_file_paths

def get_model_dirs_from(khs_label_root_dirs):
    for khs_label_root_dir in khs_label_root_dirs:
        for root,dirs,files in tqdm(os.walk(khs_label_root_dir, topdown=False)):
            if not dirs: model_dir.append(root)

def process_json(json_file_path):
    # 'polygon2' is face
    # 'polygon1' is hair
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    image_dir = os.path.dirname(json_file_path.replace('labels', 'images'))
    image_path = os.path.join(image_dir, json_data['filename'])
    try:
        h, w = int(json_data['height']), int(json_data['width'])
    except:
        assert os.path.exists(image_path)
        h, w = imagesize.get(image_path)
    polygon = json.loads(json_data.get('polygon2', '[]'))
    face_size = 0
    ratio = 0
    if polygon:
        polygon = polygon[0]
        blank = np.zeros((h, w), np.uint8)
        coords_list = []
        for pt in polygon:
            pt_coord = [int(pt['x']), int(pt['y'])]
            coords_list.append(pt_coord)
        coords = np.array(coords_list).reshape((-1, 1, 2))
        blank = cv2.fillPoly(blank, [coords], 1)
        face_size = int(np.sum(blank))
        ratio = face_size/(h*w)
    return {json_file_path: {'h':h, 'w':w, 'face_size':face_size, 'ratio':ratio, 'image_path':image_path}}

khs_label_root_dirs = ['/data_new/chence/K-Hairstyle/Validation/rawset/labels/0003.rawset', '/data_new/chence/K-Hairstyle/Training/rawset/labels/0003.rawset']
# find all json files in khs_root_dir
model_dir = []
json_file_paths = []
get_model_dirs_from(khs_label_root_dirs)
output_meta = {}

with Pool(128) as p:
    result = list(tqdm(p.imap(get_json_files_from, model_dir), total=len(model_dir)))
    for _result in result:
        json_file_paths.extend(_result)

with Pool(128) as p:
    res = list(tqdm(p.imap(process_json, json_file_paths), total=len(json_file_paths)))

output_meta = {k: v for d in res for k, v in d.items()}

with open('/home/chence/Research/3DHeadGen/DatProc/temp/KHairstyle2/khs_filtered.json', 'w') as f:
    json.dump(output_meta, f, indent=4)
