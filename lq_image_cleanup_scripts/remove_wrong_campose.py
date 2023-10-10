'''
Author: tianhao 120090472@link.cuhk.edu.cn
Date: 2023-10-10 16:44:14
LastEditors: tianhao 120090472@link.cuhk.edu.cn
LastEditTime: 2023-10-10 16:45:24
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
from tqdm import tqdm

def config_parser():
    parser = argparse.ArgumentParser(description='This script removes the wrong campose images from the dataset.json file.')
    parser.add_argument('--src_json', type=str, default='/data1/chence/PanoHeadData/single_view/dataset.json', help='The source json file.')
    parser.add_argument('--filter_view', type=str, default='back', choices=['back', 'front'], help='The view to be filtered.')
    return parser.parse_args()

args =  config_parser()
assert os.path.exists(args.src_json), f'{args.src_json} does not exist.'
save_json_path = args.src_json.replace('.json', f'_no_{args.filter_view}.json')
with open(args.src_json, 'r') as f:
    json_origin = json.load(f)
json_no_wrong_campose = {}
print(f'Before processing, there are {len(json_origin)} images.')
with open(save_json_path, 'w'):
    json.dump(json_no_wrong_campose, f, indent=2)