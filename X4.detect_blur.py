'''
Author: tianhao 120090472@link.cuhk.edu.cn
Date: 2023-10-15 11:08:21
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-10-15 15:59:22
FilePath: /DatProc/X4.detect_blur.py
Description:    

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import os
import cv2
import json
import tqdm
import shutil
import argparse
import numpy as np
from functools import partial
from multiprocessing import Pool


def get_blur_degree_svd(image:np.array, sv_num=10):
    """
    Modified from https://github.com/fled/blur_detection/blur_detection.py
    [0: clear ~ 1: blur]
    """
    u, s, v = np.linalg.svd(image)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    score = top_sv/total_sv
    return score

def get_blur_degree_laplacian(image: np.array):
    """
    Modified from https://github.com/WillBrennan/BlurDetection2/process.py
    Low: blur
    High: clear
    """
    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    return score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to json file.', type=str)
    parser.add_argument('-o', '--output', help='path to copy clear images.', type=str, default='/data2/tianhao/DAD_clear_new')
    parser.add_argument('--lap_threshold', help='threshold for laplacian blur detection.', type=int, default=50) # 50
    parser.add_argument('--svd_threshold', help='threshold for SVD blur detection.', type=int, default=0.75) # 0.75
    parser.add_argument('--force', help='force to overwrite existing files.', action='store_true')
    return parser.parse_args()

def get_images(input_path):
    process_files = []
    if input_path.endswith('.json'):
        print(f'Loading json')
        with open(args.input, 'r') as f:
            meta = json.load(f)
        print(f'Loading meta data')
        for image_rel_path in tqdm.tqdm(meta.keys()):
            process_files.append(image_rel_path)
    elif os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.endswith('.png'):
                process_files.append(file)
    else:
        process_files.append(input_path)
    return process_files

args = parse_args()
image_out_name = os.path.join(os.path.dirname(args.input), 'dataset_blur.json')
blur_image_out_name = os.path.join(os.path.dirname(args.input), f'dataset_blur_lap_{args.lap_threshold}svd_{args.svd_threshold}.json')


if os.path.exists(image_out_name) and not args.force:
    image_meta = json.load(open(image_out_name, 'r'))
else:
    image_meta = {}
    process_files = get_images(args.input)
    for file in tqdm.tqdm(process_files):
        if os.path.basename(args.input) == 'dataset.json':
            image_abs_path = os.path.join(os.path.dirname(args.input), 'align_images', file.replace('png', 'jpg'))
        else:
            image_abs_path = os.path.join(args.input, file)
        image_data = cv2.imread(image_abs_path, cv2.IMREAD_GRAYSCALE)
        svd_score = get_blur_degree_svd(image_data)
        laplacian_score = get_blur_degree_laplacian(image_data)
        image_meta[file] = {'svd_score': svd_score, 'laplacian_score': laplacian_score}
    with open(image_out_name, 'w') as f:
        json.dump(image_meta, f)

blur_image_meta = {}
svd_blur = 0
laplacian_blur = 0
both = 0
no = 0
# stat blur images
for key, value in tqdm.tqdm(image_meta.items()):
    if value['svd_score'] > args.svd_threshold:
        svd_blur += 1
    if value['laplacian_score'] < args.lap_threshold:
        laplacian_blur += 1
    if value['svd_score'] > args.svd_threshold and value['laplacian_score'] < args.lap_threshold:
        both += 1
        blur_image_meta[key] = value
    if value['svd_score'] < args.svd_threshold and value['laplacian_score'] > args.lap_threshold:
        no += 1
    # else:
    #     src_image_abs_path = os.path.join(args.input, key)
    #     dst_image_abs_path = os.path.join(args.output, key)
    #     shutil.copy(src_image_abs_path, dst_image_abs_path)
        # copy clear images
assert len(image_meta) == svd_blur + laplacian_blur - both + no
results = {
    'both': both,
    'svd_blur': svd_blur,
    'laplacian_blur': laplacian_blur,
    'no': no
}
print(f'Total: {len(image_meta)}')
for key, value in results.items():
    print(f'{key}: {value}')        
with open(blur_image_out_name, 'w') as f:
    json.dump(blur_image_meta, f)
