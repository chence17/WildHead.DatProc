'''
Author: tianhao 120090472@link.cuhk.edu.cn
Date: 2023-09-26 09:57:54
LastEditors: tianhao 120090472@link.cuhk.edu.cn
LastEditTime: 2023-09-28 16:03:32
FilePath: /DatProc/05.rebalance_stat.py
Description: 
    Code to rebalance Dataset
Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import os
import json
import argparse
import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from scipy import stats
from multiprocessing import Pool
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from utils.cam_pose_utils import get_cam_coords

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, help="path to json metafile", default="/data/PanoHeadData/single_view/dataset.json")
    parser.add_argument("-o", "--output_dir", type=str, help="path to output directory", default="./temp")
    parser.add_argument("-j", "--num_workers", type=int, help="number of workers", default=256)
    # parser.add_argument("--force", action="store_true", help="force to overwrite existing files")
    args, _ = parser.parse_known_args()
    return args

def get_coord():
    coords_save_path = os.path.join(args.output_dir, "coords.npy")
    coords = []
    if not(hasattr(args, 'force')):
        if os.path.exists(coords_save_path):
            coords = np.load(coords_save_path)

    if isinstance(coords, list):
        print(f'Calculating coords...')
        with open(args.input_path, 'r') as f:
            json_data = json.load(f)
        for img_path, img_meta in tqdm(json_data.items()):
            c2w = np.array(img_meta["camera"][:16]).reshape(4,4)
            theta, phi, r, x, y, z = get_cam_coords(c2w)
            if theta < -90 and theta >= -180: theta += 360
            coords.append((theta, phi))
        coords = np.array(coords)
        np.save(coords_save_path, coords)
    return coords

def get_density(coords: np.array):
    global kernel
    density_save_path = os.path.join(args.output_dir, "density.npy")
    densities = []
    if not(hasattr(args, 'force')):
        if os.path.exists(density_save_path):
            densities = np.load(density_save_path)

    if isinstance(densities, list):
        print(f'Calculating density...')
        kernel = stats.gaussian_kde(coords.T)
        for coord in tqdm(coords):
            density = kernel(coord)
            densities.append(density)
        densities = np.array(densities)
        print(f'Maximum density: {np.max(densities)}, Minimum density: {np.min(densities)}')
        np.save(density_save_path, densities)
    return densities

def get_num_duplicate(densities: np.array, bin_bounds: np.array, a=0.002):
    N = []
    Z = []
    for density in tqdm(densities):
        if density < bin_bounds[1]:
            _N = len(bin_bounds) - 1
        else:
            _N = min(len(bin_bounds) - 2, max(1, round(a/density)))
        N.append(_N)
        Z.append(density*_N)
    return N, Z

def main():
    global kernel

    coords = get_coord()
    densities = get_density(coords)
    
    fig, axs = plt.subplots(3,1, figsize=(5, 15))
    ax1 = axs[0]
    theta = coords[:, 0]
    phi = coords[:, 1]
    ax1.scatter(theta, phi, s=0.05, c=densities, cmap='plasma')
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$\phi$")
    ax1.set_xticks(np.arange(-90, 271, 90))
    ax1.set_yticks(np.arange(0, 181, 30))
    ax1.set_xlim(-100, 300)
    ax1.set_ylim(-20, 200)
    ax1.set_xticklabels(['-90\n(back)', '0\n(left)', '90\n(front)', '180/-180\n(right)', '-90\n(back)'])
    ax1.set_yticklabels(['0(up)', '30', '60', '90(front)', '120', '150', '180(down)'])

    ax2 = axs[1]
    densities_unique, counts = np.unique(densities, return_counts=True)
    cumulative_counts = np.cumsum(counts)
    find_percentiles = [0, 20, 40, 60, 80, 100]
    percentiles = np.percentile(densities, find_percentiles)
    data_labels = []
    ax_xrange = np.linspace(densities_unique[0], densities_unique[-1], 1000)
    for value in ax_xrange:
        for idx, _ in enumerate(percentiles[:-1]):
            if percentiles[idx] <= value <= percentiles[idx+1]:
                data_labels.append(len(find_percentiles) - idx - 1)
                break
    ax2.plot(ax_xrange, data_labels, color='r')
    ax2.set_ylabel("Number of duplicates")
    ax2.set_yticks(np.arange(1,len(find_percentiles)))
    ax3 = ax2.twinx()
    ax3.plot(densities_unique, cumulative_counts)
    ax3.set_xlabel("Probability Density")
    ax3.set_ylabel("Comulative Number of Densities")

    ax3.vlines(percentiles, np.zeros_like(percentiles), np.arange(len(find_percentiles)-1, -1, -1), colors='r', linestyles='dashed')

    ax4 = axs[2]
    N = []
    for value in densities:
        for i in range(len(percentiles)-1):
            if percentiles[i] <= value <= percentiles[i+1]:
                N.append(len(find_percentiles) - i)
                break
    N = np.array(N)
    ax4.scatter(theta, phi, s=0.05, c=N, cmap='GnBu')
    ax4.set_xlabel(r"$\theta$")
    ax4.set_ylabel(r"$\phi$")
    ax4.set_xticks(np.arange(-90, 271, 90))
    ax4.set_yticks(np.arange(0, 181, 30))
    ax4.set_xlim(-100, 300)
    ax4.set_ylim(-20, 200)
    ax4.set_xticklabels(['-90\n(back)', '0\n(left)', '90\n(front)', '180/-180\n(right)', '-90\n(back)'])
    ax4.set_yticklabels(['0(up)', '30', '60', '90(front)', '120', '150', '180(down)'])

    plt.show()

if __name__ == '__main__':
    kernel = None
    args = parse_args()
    main()