'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-10-04 20:22:28
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-10-04 21:41:15
FilePath: /DatProc/k-hairstyle_process/05.khs_crop.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import cv2
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.recrop_images import crop_final
from utils.tool import transform_box, box2quad
from utils.bv_utils import get_final_crop_size


def parse_args():
    parser = argparse.ArgumentParser(description='Filter images.')
    parser.add_argument('-i', '--json_file', type=str, help='path to the json file', required=True)
    args, _ = parser.parse_known_args()

    if not os.path.isabs(args.json_file):
        args.json_file = os.path.abspath(args.json_file)
        print("Set json file to be absolute:", args.json_file)
    assert os.path.exists(args.json_file), f'args.json_file {args.json_file} does not exist!'

    return args


def no_rotation_angle(box_np):
    x1, y1, w, h = box_np
    x2, y2 = x1 + w, y1 + h
    box_quad = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]).astype(np.float32)
    box_center = np.mean(box_quad, axis=0)
    return 0.0, box_center


def main(args):
    print(args)
    # load json file
    with open(args.json_file, 'r', encoding='utf8') as f:
        dtdict = json.load(f)

    scale = [0.7417686609206039, 0.7417686609206039]
    shift = [-0.007425799169690871, 0.00886478197975557]
    crop_size = get_final_crop_size(512)
    target_wh = (512, 512)
    save_folder = os.path.dirname(args.json_file)
    cropped_image_folder = os.path.join(save_folder, 'cropped_images')
    os.makedirs(cropped_image_folder, exist_ok=True)
    print("align_image_folder:", cropped_image_folder)
    for dtkey, dtitem in tqdm(dtdict.items()):
        head_boxes = dtitem['raw']['head_boxes']
        image_path = os.path.join(save_folder, dtitem['raw']['file_path'])
        medium_path = '/'.join(os.path.dirname(dtitem['raw']['file_path']).split('/')[1:])
        image_name = os.path.basename(image_path)[:-4]
        image_data = cv2.imread(image_path)
        for box_id, box in head_boxes.items():
            try:
                cropped_img, _, _ = crop_final(
                    image_data.copy(), size=crop_size,
                    quad=box2quad(transform_box(box, scale, shift)),
                    top_expand=0., left_expand=0.,
                    bottom_expand=0., right_expand=0.
                )
                cropped_img = cv2.resize(cropped_img, target_wh, interpolation=cv2.INTER_BICUBIC)
                cropped_image_path = os.path.join(cropped_image_folder, medium_path, f"{image_name}_{box_id}.png")
                os.makedirs(os.path.dirname(cropped_image_path), exist_ok=True)
                cv2.imwrite(cropped_image_path, cropped_img)
            except:
                print(f"Error in {image_path} {box_id}")


if __name__ == '__main__':
    # Camera Checked.
    args = parse_args()
    main(args)

