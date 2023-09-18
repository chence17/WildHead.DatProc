import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from utils.dataset_process import find_meta_files

def parse_args():
    parser = argparse.ArgumentParser(description='Run Frontal View Pipeline')
    parser.add_argument('json_dir', type=str, help='path to json file directory',
                        default='/home/shitianhao/project/DatProc/utils/stats/')
    args, _ = parser.parse_known_args()
    return args

def main(args):
    h2b_ratios = []
    json_file_paths = find_meta_files(args.json_dir)
    for json_file_path in tqdm(json_file_paths, position=0, leave=True):
        with open(json_file_path, 'r') as f:
            meta = json.load(f)
        for img_rel_path, img_meta in meta.items():
            for box_id, box_meta in (img_meta.items()):
                if box_meta['frontal'] is False: continue
                h2b_ratios.append(box_meta['h2b_ratio'])
    h2b_ratios = np.array(h2b_ratios)
    print(f'mean: {np.mean(h2b_ratios)}, std: {np.std(h2b_ratios)}')

if __name__ == '__main__':
    args = parse_args()
    main(args)