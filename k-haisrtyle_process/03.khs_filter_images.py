'''
This script is modified from 01.filter_images.py to filter small images from the back-view only k-hairstyle dataset.
'''
import os
import cv2
import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool

from utils.filter import load_image_names, filter_invalid_and_small
from utils.head_detection import YoloHeadDetector
from utils.tool import partition_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Filter images.')
    parser.add_argument('-i', '--image_folder', type=str, help='path to the folder that holds images', required=True)
    parser.add_argument('-d', '--data_source', type=str, help='data source, eg: web/data', required=True)
    parser.add_argument('-j', '--num_processes', type=int, help='number of processes (default 64)', default=64)
    parser.add_argument('-p', '--partition_size', type=int, help='number of images in a partition of datset (default 10000)', default=10000)
    args, _ = parser.parse_known_args()

    if not os.path.isabs(args.image_folder):
        args.image_folder = os.path.abspath(args.image_folder)
        print("Set image folder to be absolute:", args.image_folder)
    assert os.path.exists(args.image_folder), f'args.image_folder {args.image_folder} does not exist!'
    assert args.num_processes > 0, f'args.num_processes {args.num_processes} must be positive integer'

    return args


def main(args):
    print(args)
    # initialize detectors
    hbox_det = YoloHeadDetector(weights_file='assets/224x224_yolov4_hddet_480x640.onnx',
                                input_width=640, input_height=480)

    # load image folder and save folder
    image_folder = args.image_folder
    assert os.path.exists(image_folder), f'image directory {image_folder} does not exist!'
    image_folder_name = os.path.basename(image_folder)
    save_folder = os.path.dirname(image_folder)
    print("image_folder_name:", image_folder_name)
    print("save_folder:", save_folder)

    # search through dataset dir and get all image formats
    print("loading images...")
    image_paths = load_image_names(image_folder)
    meta_dict = {}

    # filtering invalid and small images
    print("filtering invalid and small images...")
    with Pool(processes=args.num_processes) as pool:
        mp_results = list(tqdm(pool.imap(filter_invalid_and_small, image_paths), total=len(image_paths)))
    image_paths = [res for res in mp_results if res is not None]

    print("detecting head boxes...")
    for image_path in tqdm(image_paths):
        image_data = cv2.imread(image_path)
        head_boxes = hbox_det(image_data.copy(), isBGR=True)
        if head_boxes is None or head_boxes.shape[0] == 0:
            continue
        image_path = os.path.relpath(image_path, os.path.realpath(save_folder))
        meta_dict[image_path] = {
            'data_source': args.data_source,
            'raw': {
                'file_path': image_path,
                'head_boxes': {}
            }
        }
        for idx, box in enumerate(head_boxes):
            meta_dict[image_path]['raw']['head_boxes'][f'{idx:02d}'] = box.tolist()

    print("saving meta files...")
    total_partition = len(meta_dict) // args.partition_size + 1
    for idx, part_dict in tqdm(enumerate(partition_dict(meta_dict, args.partition_size), 1)):
        out_json_name = f'meta_{idx}-{total_partition}.json'
        out_json_path = os.path.join(save_folder, out_json_name)
        with open(out_json_path, 'w', encoding='utf8') as f:
            json.dump(part_dict, f, indent=4)


if __name__ == '__main__':
    # Box Checked.
    args = parse_args()
    main(args)
