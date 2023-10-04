'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-09-30 17:18:07
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-09-30 17:21:30
FilePath: /DatProc/k-hairstyle_process/x1.split_khs_meta.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import json
import argparse
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tool import partition_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Split meta files.')
    parser.add_argument('-i', '--json_file', type=str, help='path to the json file', required=True)
    parser.add_argument('-n', '--num_partition', type=int, help='number of partitions', required=True)
    return parser.parse_args()

def main(args):
    with open(args.json_file, 'r', encoding='utf8') as f:
        meta_dict = json.load(f)

    print("saving meta files...")
    save_folder = os.path.dirname(args.json_file)
    total_partition = args.num_partition
    dm = divmod(len(meta_dict), total_partition)
    partition_size = dm[0] + (dm[1] != 0)
    for idx, part_dict in tqdm(enumerate(partition_dict(meta_dict, partition_size), 1)):
        out_json_name = f'meta_{idx}-{total_partition}.json'
        out_json_path = os.path.join(save_folder, out_json_name)
        with open(out_json_path, 'w', encoding='utf8') as f:
            json.dump(part_dict, f, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main(args)