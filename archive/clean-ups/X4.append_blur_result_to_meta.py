'''
Author: tianhao 120090472@link.cuhk.edu.cn
Date: 2023-10-22 21:20:13
LastEditors: tianhao 120090472@link.cuhk.edu.cn
LastEditTime: 2024-02-25 20:59:26
FilePath: /DatProc/X4.append_blur_result_to_meta.py
Description: Append the blur ratio to meta file

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
import os
import json
import tqdm

blur_det_result_meta_path = '/data/PanoHeadData/single_view_hq/dataset_blur.json'
origin_meta_path = '/data/PanoHeadData/single_view_hq/dataset.json'


print(f'Loading {blur_det_result_meta_path}...')
with open(blur_det_result_meta_path, 'r') as f:
    blur_det_result_meta = json.load(f)
print(f'Loading {origin_meta_path}...')
with open(origin_meta_path, 'r') as f:
    origin_meta = json.load(f)

for img_key, img_val in tqdm.tqdm(blur_det_result_meta.items()):
    assert img_key in origin_meta.keys()
    origin_meta[img_key].update(img_val)

print(f'Saving to {origin_meta_path}...')
with open(origin_meta_path, 'w') as f:
    json.dump(origin_meta, f, indent=4)
