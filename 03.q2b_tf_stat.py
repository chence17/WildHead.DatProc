import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from utils.process_utils import find_meta_files

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_dir', type=str, help='Directory containing json files')
    raw_args = parser.parse_args()
    raw_args.json_dir = os.path.realpath(raw_args.json_dir)
    return raw_args

def main(args):
    meta_file_paths = find_meta_files(args.json_dir)
    dataset_scales = []
    dataset_shifts = []
    num_imgs = 0
    for meta_file_path in tqdm(meta_file_paths, position=0, leave=True):
        with open(meta_file_path, 'r') as f:
            meta = json.load(f)
        for img_meta in tqdm(meta.values(), position=1, leave=False):
            raw_meta = img_meta['raw']
            img_scales = []
            img_shifts = []
            for box_q2b_val in raw_meta['q2b_tf'].values():
                if box_q2b_val is None: continue
                num_imgs += 1
                scale = box_q2b_val['scale']
                shift = box_q2b_val['shift']
                img_scales.append(scale)
                img_shifts.append(shift)
            dataset_scales.extend(img_scales)
            dataset_shifts.extend(img_shifts)
    dataset_scales = np.array(dataset_scales)
    dataset_shifts = np.array(dataset_shifts)
    average_scale = np.mean(dataset_scales, axis=0)
    average_shift = np.mean(dataset_shifts, axis=0)
    print(f'Average scale: {average_scale}; Average shift: {average_shift}')
    print(f'Number of samples: {num_imgs}')
    

    
def process_meta_file(meta_file):
    meta_scales = []
    meta_shifts = []
    for img_meta in meta_file.values():
        raw_meta = img_meta['raw']
        img_scales = []
        img_shifts = []
        for box_q2b_val in raw_meta['q2b_tf'].values():
            if box_q2b_val is None: continue
            scale = box_q2b_val['scale']
            shift = box_q2b_val['shift']
            img_scales.append(scale)
            img_shifts.append(shift)
        meta_scales.extend(img_scales)
        meta_shifts.extend(img_shifts)
    return meta_scales, meta_shifts

if __name__ == '__main__':
    args = parse_args()
    main(args)
