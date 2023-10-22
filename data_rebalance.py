import os
import json
import tqdm
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Rebalance dataset')
    parser.add_argument('-i', '--input_path', type=str, help='path to json metafile', default='/data2/chence/single_view_hq/dataset.json')
    parser.add_argument('-d', '--dict_file', type=str, default='/home/shitianhao/project/DatProc/temp/dup_per_deg.json')
    parser.add_argument('-o', '--output_path', type=str, help='path to save rebalanced dataset', default='/data2/chence/single_view_hq/dataset.json')
    parser.add_argument('-n', '--num_worker', type=int, help='number of workers', default=64)
    return parser.parse_args()

def main(args):

    with open(args.input_path, 'r') as f:
        dataset = json.load(f)

    with open(args.dict_file, 'r') as f:
        dup_per_deg = json.load(f)

    for image_name, image_meta in tqdm.tqdm(dataset.items()):
        camera_scoord = image_meta['camera_scoord']
        theta = int(camera_scoord[0])
        dup_num = dup_per_deg[str(theta)]
        dataset[image_name]['dup_num'] = dup_num

    with open(args.output_path, 'w') as f:
        json.dump(dataset, f, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main(args)
