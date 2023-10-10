'''
Author: tianhao 120090472@link.cuhk.edu.cn
Date: 2023-10-10 16:24:13
LastEditors: tianhao 120090472@link.cuhk.edu.cn
LastEditTime: 2023-10-10 16:41:54
FilePath: /DatProc/lq_image_cleanup_scripts/delete_dad.py
Description: This script is used to delete the entries belonging to DAD-3DHeads dataset from the dataset.json file.

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''

import json
from tqdm import tqdm

SRC_JSON = '/data1/chence/PanoHeadData/single_view/dataset.json'
DST_JSON = '/data1/chence/PanoHeadData/single_view/dataset_no_dad.json'

with open(SRC_JSON, 'r') as f:
    json_origin = json.load(f)
json_no_dad = {}
print(f'Before processing, there are {len(json_origin)} images.')
for img, img_meta in tqdm(json_origin.items()):
    if not(img_meta['data_source'].startswith('DAD-3DHeads')):
        json_no_dad[img] = img_meta
print(f'After processing, there are {len(json_no_dad)} images left.')
with open(DST_JSON, 'w') as f:
    json.dump(json_no_dad, f, indent=2, sort_keys=True)