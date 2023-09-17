import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm

from recrop_images import Recropper
from utils.FaceParsing import HeadParser
from utils.dataset_process import FaceAlignmentDetector, ProcessError, calc_h2b_ratio


def parse_args():
    parser = argparse.ArgumentParser(description='Run Frontal View Pipeline')
    parser.add_argument('file', type=str, help='path to json file',
                        default='/home/shitianhao/project/DatProc/utils/stats/mh_dataset_stat.json')
    # parser.add_argument('-j', '--num_processes', type=int, help='number of processes', default=32)
    args, _ = parser.parse_known_args()
    return args

def process_image(img_path, meta, fdet, recropper, hpar, save_img_dir, save_sem_dir):
    img = cv2.imread(img_path)
    for box_id, box_meta in tqdm(meta.items(), position=1, leave=False):
        box_x, box_y, box_w, box_h = box_meta['head_box']
        box_image = img[int(box_y):int(box_y+box_h),
                        int(box_x):int(box_x+box_w)].copy()
        try:
            landmarks = fdet(box_image, True)
        except ProcessError:
            box_meta['landmarks'] = None
            box_meta['frontal'] = False
        else:
            box_meta['landmarks'] = landmarks.tolist()
            box_meta['frontal'] = True
        try:
            if box_meta['frontal'] == False: raise ProcessError
            cropped_img, camera_poses, quad = recropper(box_image, landmarks)
        except ProcessError:
            box_meta['frontal'] = False
            box_meta['camera_poses'] = None
            box_meta['quad'] = None
        else:
            img_name = os.path.basename(img_path)
            save_img_path = os.path.join(
                save_img_dir, f'{img_name[:-4]}_{box_id}{img_name[-4:]}')
            cv2.imwrite(save_img_path, cropped_img)
            sem = hpar(cropped_img, is_bgr=True)
            mask = (sem != 0).astype(np.uint8)
            save_sem_path = os.path.join(
                save_sem_dir, f'{img_name[:-4]}_{box_id}{img_name[-4:]}')
            cv2.imwrite(save_sem_path, sem)
            box_meta['camera_poses'] = camera_poses.tolist()
            box_meta['quad'] = quad.tolist()
            box_meta['h2b_ratio'] = calc_h2b_ratio(mask)
        meta[box_id] = box_meta


def main(args):
    # initialize face detecor and recropper
    fdet = FaceAlignmentDetector()
    recropper = Recropper()
    hpar = HeadParser()
    # load metadata file
    with open(args.file, 'r') as f:
        data = json.load(f)
    # output dir
    json_dir = os.path.dirname(args.file)
    cropped_img_dir = os.path.join(json_dir, 'cropped_images')
    cropped_sem_dir = os.path.join(json_dir, 'cropped_semantic')
    os.makedirs(cropped_img_dir, exist_ok=True)
    os.makedirs(cropped_sem_dir, exist_ok=True)
    for img_path, meta in tqdm(data.items(), position=0, leave=True):
        abs_img_path = os.path.join(json_dir, img_path)
        process_image(abs_img_path, meta, fdet, recropper, hpar, cropped_img_dir, cropped_sem_dir)
    # save metadata file
    with open(args.file, 'w') as f:
        json.dump(data, f, indent=4)



if __name__ == '__main__':
    args = parse_args()
    main(args)