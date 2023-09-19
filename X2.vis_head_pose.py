'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-09-19 22:51:33
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-09-19 23:04:13
FilePath: /DatProc/X2.vis_head_pose.py
Description: X2.vis_head_pose.py
'''
import os
import cv2
import json
import numpy as np
import argparse

from utils.face_parsing import show_image
from utils.tool import render_camera

def parse_args():
    parser = argparse.ArgumentParser(description='Filter images.')
    parser.add_argument('-i', '--json_file', type=str, help='path to the json file', required=True)
    parser.add_argument('-d', '--data_item', type=str, help='data item', default=None)
    args, _ = parser.parse_known_args()

    if not os.path.isabs(args.json_file):
        args.json_file = os.path.abspath(args.json_file)
        print("Set json file to be absolute:", args.json_file)
    assert os.path.exists(args.json_file), f'args.json_file {args.json_file} does not exist!'

    return args


def main(args):
    print(args)
    # load json file
    with open(args.json_file, 'r', encoding='utf8') as f:
        dtdict = json.load(f)

    root_folder = os.path.dirname(args.json_file)

    if args.data_item is None:
        print("data item is None, using the first data item in json file")
        args.data_item = list(dtdict.keys())[0]
    print("data item:", args.data_item)
    dt = dtdict[args.data_item]
    for box_id, box_dt in dt['head'].items():
        cam = box_dt['camera']
        r_img = render_camera(cam)
        h, w = r_img.shape[:2]
        a_img = cv2.imread(os.path.join(root_folder, box_dt['align_image_path']))
        a_img = cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB)
        a_img = cv2.resize(a_img, (h, w))
        vis_img = np.hstack([r_img, a_img])
        show_image(vis_img, is_bgr=False, title=dt['raw']['file_path']+f"_{box_id}", show_axis=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)
