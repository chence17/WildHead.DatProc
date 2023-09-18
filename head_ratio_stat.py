import os
import json
import argparse
from tqdm import tqdm

from utils.dataset_process import find_meta_files

def parse_args():
    parser = argparse.ArgumentParser(description='Run Frontal View Pipeline')
    parser.add_argument('json_dir', type=str, help='path to json file directory',
                        default='/home/shitianhao/project/DatProc/utils/stats/')
    # parser.add_argument('-j', '--num_processes', type=int, help='number of processes', default=32)
    args, _ = parser.parse_known_args()
    return args

def main(args):
    h2b_ratios = []
    json_file_paths = find_meta_files(args.json_dir)
    for json_file_path in json_file_paths:
        with open(json_file_path, 'r') as f:
            meta = json.load(f)
        for img_rel_path, img_meta in meta.items():
            img_abs_path = os.path.join(args.json_dir, img_rel_path)
            for box_id, box_meta in img_meta.items():
                h2b_ratios.append(box_meta['h2b_ratio'])
        pass

if __name__ == '__main__':
    args = parse_args()
    main(args)