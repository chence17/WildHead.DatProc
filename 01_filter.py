import os
import cv2
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from utils.dataset_process import get_images, YoloHeadDetector, get_all_extensions

def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Statistic')
    parser.add_argument('-d', '--dataset_path', type=str, help='path to dataset', default='/home/shitianhao/project/DatProc/assets/mh_dataset')
    parser.add_argument('-o', '--output_dir', type=str, help='path to output dir', default='/home/shitianhao/project/DatProc/utils/stats')
    parser.add_argument('-j', '--num_processes', type=int, help='number of processes', default=32)
    args, _ = parser.parse_known_args()
    return args

is_small = lambda w, h, min_size=512: w < min_size or h < min_size

def detect_head(img_path, hdet):
    img = cv2.imread(img_path)
    if img is None: return # filter invalid images
    img_h, img_w = img.shape[:2]
    if is_small(img_h, img_w): return # filter small images
    image_boxes = hdet(img.copy(), isBGR=True)
    if image_boxes is None or image_boxes.shape[0] == 0: return # filter images without boxes
    return image_boxes

def format_headbox(img_boxes):
    hbox_dict = {}
    for idx, box in enumerate(img_boxes):
        hbox_dict[f'{idx:02d}'] = box.tolist()
    return hbox_dict

def process(img_path, dataset_path, hdet):
    img_boxes = detect_head(img_path, hdet)
    if img_boxes is None: return img_path, None
    save_path = os.path.relpath(img_path, dataset_path)
    return save_path, format_headbox(img_boxes) 

def main(args):
    # initialize detectors
    hdet = YoloHeadDetector(weights_file='assets/224x224_yolov4_hddet_480x640.onnx',
                            input_width=640, input_height=480)

    dataset_path = args.dataset_path
    assert os.path.exists(dataset_path), 'data path does not exist'

    # output configs
    data_name = os.path.basename(dataset_path)
    out_json_name = data_name + '_stat.json'
    out_json_path = os.path.join(args.output_dir, out_json_name)

    # search through dataset dir and get all image formats
    extensions = get_all_extensions(dataset_path)
    print(f'Found extensions: {extensions}')
    IMG_FORMATS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".svg", ".ico", ".exif", ".raw", ".heic", ".jfif", ".tga", ".pdf", ".eps", ".ai", ".psd"]
    extensions = list(set(extensions).intersection(set(IMG_FORMATS)))
    print(f'Found image extensions: {extensions}')

    img_paths = get_images(dataset_path, extensions)
    meta_dict = {}

    with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
        results = list(tqdm(executor.map(lambda x: process(x,dataset_path, hdet), img_paths), total=len(img_paths)))

    for img_path, hbox_dict in tqdm(results):
        if hbox_dict is None: continue
        meta_dict[img_path] = hbox_dict

    with open(out_json_path, 'w') as f:
        json.dump({os.path.realpath(args.dataset_path): meta_dict}, f, indent=4)

        


    
if __name__ == '__main__':
    args = parse_args()
    main(args)
