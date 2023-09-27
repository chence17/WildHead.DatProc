'''
Author: tianhao 120090472@link.cuhk.edu.cn
Date: 2023-09-26 09:57:54
LastEditors: tianhao 120090472@link.cuhk.edu.cn
LastEditTime: 2023-09-27 10:34:11
FilePath: /DatProc/05.rebalance_stat.py
Description: 
    Code to rebalance Dataset
Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from scipy import stats
from multiprocessing import Pool
from matplotlib import pyplot as plt
from KDEpy import FFTKDE

from utils.process_utils import find_meta_files
from utils.cam_pose_utils import get_cam_coords

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--json_dir", type=str, help="path to json metafile directory", default="./temp/ffhq_meta")
    parser.add_argument("-o", "--output_dir", type=str, help="path to output directory", default="./temp")
    parser.add_argument("-j", "--num_workers", type=int, help="number of workers", default=128)
    parser.add_argument("-f", "--force", action="store_true", help="force to overwrite existing files")
    return parser.parse_args()

def get_density(coord):
    theta, phi = coord
    density = kernel([theta, phi])[0]
    return density

def main(args):
    global kernel
    coords = []
    densities = []
    coords_save_path = os.path.join(args.output_dir, "coords.npy")
    density_save_path = os.path.join(args.output_dir, "density.npy")

    if not args.force:
        if os.path.exists(coords_save_path):
            coords = np.load(coords_save_path)
        if os.path.exists(density_save_path):
            densities = np.load(density_save_path)
    
    if isinstance(coords, list):
        print(f'Calculating coords...')
        json_file_paths = find_meta_files(args.json_dir)
        for json_file_path in tqdm(json_file_paths, position=0, leave=True):
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
            for img_path, img_meta in tqdm(json_data.items(), position=1, leave=False):
                for box_idx, box_meta in img_meta["head"].items():
                    c2w = np.array(box_meta["camera"][:16]).reshape(4,4)
                    theta, phi, r, x, y, z = get_cam_coords(c2w)
                    if theta < -90 and theta >= -180: theta += 360
                    coords.append((theta, phi))
        coords = np.array(coords)
        np.save(coords_save_path, coords)

    if isinstance(densities, list):
        print(f'Calculating density...')
        kernel = stats.gaussian_kde(coords.T)
        with Pool(args.num_workers) as pool:
            densities = list(tqdm(pool.imap(get_density, coords), total=len(coords)))
        densities = np.array(densities)
        print(f'Maximum density: {np.max(densities)}, Minimum density: {np.min(densities)}')
        np.save(density_save_path, densities)

    fig, ax = plt.subplots()
    ax.scatter(coords[:, 0], coords[:, 1], s=1)
    ax.set_xlabel("theta")
    ax.set_ylabel("phi")
    plt.show()



if __name__ == '__main__':
    kernel = None
    args = parse_args()
    main(args)