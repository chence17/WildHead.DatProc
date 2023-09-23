"""
Following the previous step, this script reads the json and copies the filterd images to a new directory
"""
import os
import json
import shutil
import logging
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--json_file', type=str, default='k-haisrtyle_filter/khs_filter.json', help='json file path')
    parser.add_argument('-o', '--output', type=str, help='output json file path', default='/datas/K-Hairstyle-Filtered')
    parser.add_argument('-j', '--num_jobs', type=int, default=128, help='number of jobs')
    return parser.parse_args()

def setup_logger():
    # use filelogger
    logger = logging.getLogger('k-haisrtyle_filter')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('k-haisrtyle_filter/khs_copy.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def main(args, logger:logging.Logger):
    with open(args.json_file, 'r') as f:
        json_data = json.load(f)
    for json_path, img_meta in tqdm(json_data.items()):
        if img_meta['face_size'] != 0: continue
        save_img_path = img_meta['image_path'].replace('/datas/K-Hairstyle', args.output)
        if os.path.exists(save_img_path): continue
        os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
        try:
            shutil.copy(img_meta['image_path'], save_img_path)
        except OSError:
            logger.error(f'Error: {img_meta["image_path"]}')


if __name__ == '__main__':
    args = parse_args()
    logger = setup_logger()
    main(args, logger)
