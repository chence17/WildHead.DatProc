'''
Author: tianhao 120090472@link.cuhk.edu.cn
Date: 2023-11-11 14:22:01
LastEditors: tianhao 120090472@link.cuhk.edu.cn
LastEditTime: 2024-02-28 09:47:28
FilePath: /DatProc/data_rebalance.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import os
import json
import tqdm
import argparse
import numpy as np
from math import floor

def parse_args():
    parser = argparse.ArgumentParser(description='Rebalance dataset')
    parser.add_argument('-i', '--input_path', type=str, help='path to json metafile', default='/data2/PanoHeadData/single_view_hq/dataset_v3.json')
    parser.add_argument('-d', '--dict_file', type=str, default='/home/shitianhao/project/DatProc/temp/dup_per_deg_test.json')
    parser.add_argument('-o', '--output_path', type=str, help='path to save rebalanced dataset', default='/data2/PanoHeadData/single_view_hq/dataset_v3_balanced.json')
    parser.add_argument('-n', '--num_worker', type=int, help='number of workers', default=64)
    return parser.parse_args()

def find_closest_group(deg):
    return deg - deg % 5


def main(args):

    with open(args.input_path, 'r') as f:
        dataset = json.load(f)

    with open(args.dict_file, 'r') as f:
        dup_per_deg = json.load(f)

    before_total_num = 0
    for image_name, image_meta in tqdm.tqdm(dataset.items()):
        before_total_num += image_meta['dup_num']
    print(f'Before modify: Total number of images: {len(dataset)}, Total number of images considering duplicate: {before_total_num}')

    during_num = 0
    for image_name in tqdm.tqdm(dataset.keys()):
        camera_scoord = dataset[image_name]['camera_scoord']
        theta = floor(camera_scoord[0])
        dup_num = dup_per_deg[str(theta)]
        during_num += dup_num
        dataset[image_name]['dup_num'] = dup_num
    print(f'During modify: Total number of images: {len(dataset)}, Total number of images considering duplicate: {during_num}')

    total_num = 0
    for image_name, image_meta in tqdm.tqdm(dataset.items()):
        total_num += image_meta['dup_num']

    print(f'Total number of images: {len(dataset)}, Total number of images considering duplicate: {total_num}')

    # with open(args.output_path, 'w') as f:
    #     json.dump(dataset, f, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main(args)
