import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from utils.dataset_process import FaceAlignmentDetector, ProcessError
from recrop_images import Recropper

def parse_args():
    parser = argparse.ArgumentParser(description='Run Frontal View Pipeline')
    parser.add_argument('-f', '--file', type=str, help='path to json success_metadata', default='/home/shitianhao/project/DatProc/utils/stats/mh_dataset_stat.json')
    parser.add_argument('-o', '--output_dir', type=str, help='path to output dir', default='/home/shitianhao/project/DatProc/assets/outputs')
    parser.add_argument('-j', '--num_processes', type=int, help='number of processes', default=32)
    args, _ = parser.parse_known_args()
    return args

def process_image(img_path, image_boxes, dataset_path, dataset_image_out_dir, fdet, recropper):
    img = cv2.imread(os.path.join(dataset_path, img_path))
    success_boxes_meta = {}
    fail_boxes_list = []
    for box_id, box in tqdm(image_boxes.items(), position=1, leave=False):
        try:
            box_out_dir = os.path.join(dataset_image_out_dir, f'{img_path[:-4]}_{box_id}{img_path[-4:]}')
            box_x, box_y, box_w, box_h = box
            box_image = img[int(box_y):int(box_y+box_h), int(box_x):int(box_x+box_w)].copy()
            landmarks = fdet(box_image, True)
            if landmarks is None: raise ProcessError  # Filter boxes with failed face detection
            cropped_img, camera_poses, quad = recropper(box_image, landmarks)
            camera_poses, quad = camera_poses.tolist(), quad.tolist()
            cv2.imwrite(box_out_dir, cropped_img)
            success_boxes_meta[box_id] = {'camera_poses': camera_poses, 'quad': quad}
        except ProcessError:
            fail_boxes_list.append(box_id)
    return img_path, success_boxes_meta, image_boxes

def main(args):
    # initialize face detecor and recropper
    fdet = FaceAlignmentDetector()
    recropper = Recropper()
    # load metadata file
    with open(args.file, 'r') as f:
        data = json.load(f)
    # config outputs
    dataset_path = list(data.keys())[0]
    dataset_name = os.path.basename(dataset_path)
    dataset_out_dir = os.path.join(args.output_dir, dataset_name)
    dataset_image_out_dir = os.path.join(dataset_out_dir, 'images')
    os.makedirs(dataset_out_dir, exist_ok=True)
    os.makedirs(dataset_image_out_dir, exist_ok=True)
    success_metadata_file_path = os.path.join(dataset_out_dir, f'{dataset_name}_success_metadata.json')
    failed_metadata_file_path = os.path.join(dataset_out_dir, f'{dataset_name}_failed_metadata.json')

    success_metadata = {}
    failed_metadata = {}
    input_images_meta = data[dataset_path].items()
    with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
            futures = []

            for img_path, image_boxes in input_images_meta:
                future = executor.submit(process_image, img_path, image_boxes, dataset_path, dataset_image_out_dir, fdet, recropper)
                futures.append(future)

            for future in tqdm(futures, position=0, leave=True):
                img_path, success_boxes_meta, image_boxes = future.result()
                if len(success_boxes_meta.keys()) > 0:
                    success_metadata[img_path] = success_boxes_meta
                else:
                    failed_metadata[img_path] = image_boxes

    with open(success_metadata_file_path, 'w') as f:
        json.dump({dataset_image_out_dir:success_metadata}, f, indent=4)
    with open(failed_metadata_file_path, 'w') as f:
        json.dump({dataset_path:failed_metadata}, f, indent=4)

                
if __name__ == '__main__':
    args = parse_args()
    main(args)