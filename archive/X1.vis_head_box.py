import os
import cv2
import json
import argparse

from utils.face_parsing import show_image
from utils.tool import draw_detection_box

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
    raw_image_file = os.path.join(root_folder, dt['raw']['file_path'])
    raw_image_data = cv2.imread(raw_image_file)
    head_boxes = dt['raw']['head_boxes']
    for k, v in head_boxes.items():
        raw_image_data = draw_detection_box(raw_image_data, v, k)
    show_image(raw_image_data, is_bgr=True, title=dt['raw']['file_path'], show_axis=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)
