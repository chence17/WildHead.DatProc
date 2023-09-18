import os
import cv2
import json
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
        for root,dirs,files in os.walk(khs_label_root_dir):
            if not dirs: model_dir.append(root)

def process_json(json_file_path):
    # 'polygon2' is face
    # 'polygon1' is hair
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    try:
        h, w = int(json_data['height']), int(json_data['width'])
        polygon = json.loads(json_data['polygon2'])
        if polygon: 
            polygon = polygon[0]
            blank = np.zeros((h, w), np.uint8)
            coords_list = []
            for pt in polygon:
                pt_coord = [int(pt['x']), int(pt['y'])]
                coords_list.append(pt_coord)
            coords = np.array(coords_list).reshape((-1, 1, 2))
            blank = cv2.fillPoly(blank, [coords], 1)
            face_size = np.sum(blank)
            output_meta[json_file_path] = {'face_size':face_size, 'ratio':face_size/(h*w)}
        else:
            output_meta[json_file_path] = {'face_size':0, 'ratio':0}
    except:
        print(json_file_path)
        output_meta[json_file_path] = {'face_size':-1, 'ratio':-1}

khs_label_root_dirs = ['/datas/K-Hairstyle/Validation/rawset/labels/0003.rawset', '/datas/K-Hairstyle/Training/rawset/labels/0003.rawset']
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
    list(tqdm(p.imap(process_json, json_file_paths), total=len(json_file_paths)))

with open('utils/khs_filter.json', 'w') as f:
    json.dump(output_meta, f, indent=4)
