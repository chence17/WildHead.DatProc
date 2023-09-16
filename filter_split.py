import os
import cv2
import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool

from utils.dataset_process import get_images, YoloHeadDetector

def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Statistic')
    parser.add_argument('image_path', type=str, help='path to the folder that holds images')
    parser.add_argument('-j', '--num_processes', type=int, help='number of processes', default=32)
    parser.add_argument('-p', '--partition_size', type=int, help='number of images in a partition of datset', default=10000)
    args, _ = parser.parse_known_args()
    return args

is_small = lambda w, h, min_size=512: w < min_size or h < min_size

def filter_invalid_and_small(img_path):
    img = cv2.imread(img_path)
    if img is None: return # filter invalid images
    img_h, img_w = img.shape[:2]
    if is_small(img_h, img_w): return # filter small images
    return img_path

def detect_head(img_path, hdet):
    img = cv2.imread(img_path)
    head_boxes = hdet(img.copy(), isBGR=True)
    if head_boxes is None or head_boxes.shape[0] == 0: return # filter images without boxes
    head_boxes_dict = format_headbox(head_boxes)
    return head_boxes_dict

def format_headbox(img_boxes):
    hbox_dict = {}
    for idx, box in enumerate(img_boxes):
        hbox_dict[f'{idx:02d}'] = {'head_box':box.tolist()}
    return hbox_dict

def partition_dict(d, partition_size):
    for i in range(0, len(d), partition_size):
        yield dict(list(d.items())[i:i + partition_size])

def main(args):
    # initialize detectors
    hdet = YoloHeadDetector(weights_file='assets/224x224_yolov4_hddet_480x640.onnx',
                            input_width=640, input_height=480)

    # input configs
    image_path = args.image_path
    assert os.path.exists(image_path), 'image directory does not exist'

    # output configs
    output_dir = os.path.dirname(image_path)

    # search through dataset dir and get all image formats
    img_paths = get_images(image_path)
    meta_dict = {}

    print("filtering invalid and small images")
    with Pool(processes=args.num_processes) as pool:
        valid_img_paths = pool.map(filter_invalid_and_small, img_paths)
    valid_img_paths = [path for path in valid_img_paths if path is not None]

    print("detecting head boxes")
    for img_path in tqdm(valid_img_paths):
        head_boxes = detect_head(img_path, hdet)
        img_path = os.path.relpath(img_path, os.path.realpath(output_dir))
        if head_boxes: meta_dict[img_path] = head_boxes

    print("saving meta files")
    total_partition = len(meta_dict) // args.partition_size + 1
    for idx, part_dict in tqdm(enumerate(partition_dict(meta_dict, args.partition_size),1)):
        out_json_name = f'meta_{idx}-{total_partition}.json'
        out_json_path = os.path.join(output_dir, out_json_name)
        with open(out_json_path, 'w') as f:
            json.dump(part_dict, f, indent=4)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
