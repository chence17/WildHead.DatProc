import os
import json
import argparse
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Create partition')
    parser.add_argument('input_file', type=str, help='input json path')
    parser.add_argument('partition', type=str, help='which angle of face to keep. could be either one of "front", "left", "right",  "back" or two numbers joined by comma, e.g. "0,1"')
    return parser.parse_args()

def get_cam_coords(c2w):
    # Copied from datproc_v1.py
    # World Coordinate System: x(right), y(up), z(forward)
    T = c2w[:3, 3]
    x, y, z = T
    r = np.sqrt(x**2+y**2+z**2)
    # theta = np.rad2deg(np.arctan(np.sqrt(x**2+z**2)/y))
    theta = np.rad2deg(np.arctan2(x, z))
    if theta >= -90 and theta <= 90:
        theta += 90
    elif theta >= -180 and theta < -90:
        theta += 90
    elif theta > 90 and theta <= 180:
        theta -= 270
    else:
        raise ValueError('theta out of range')
    # phi = np.rad2deg(np.arctan(z/x))+180
    phi = np.rad2deg(np.arccos(y/r))
    return [theta, phi, r, x, y, z]  # [:3] sperical cood, [3:] cartesian cood

def get_position(raw_pos_arg):
    raw_pos_arg = raw_pos_arg.lower()
    if raw_pos_arg == 'left':
        return 135, -135
    elif raw_pos_arg == 'right':
        return -45, 45
    elif raw_pos_arg == 'front':
        return 45, 135
    elif raw_pos_arg == 'back':
        return -135, -45
    elif ',' in raw_pos_arg:
        try:
            a, b = raw_pos_arg.split(',')
            a = int(a)
            b = int(b)
            return a, b
        except:
            raise RuntimeError('Invalid partition argument')
    else:
        raise RuntimeError('Invalid partition argument')

def _is_valid_cross_180(image_degree, min_angle, max_angle):
    # in this case min_angle is larger than max_angle. e.g. for 'left', min_angle = 135, max_angle = -135
    assert min_angle > max_angle and image_degree <= 180 and image_degree >= -180
    return image_degree <= max_angle or image_degree >= min_angle

def _is_valid(image_degree, min_angle, max_angle):
    return min_angle <= image_degree and image_degree <= max_angle

def main():
    args = parse_args()
    partition = get_position(args.partition) # [min_angle, max_angle]
    is_valid = _is_valid_cross_180 if partition[0] > partition[1] else _is_valid
    input_file_path = args.input_file
    assert os.path.exists(input_file_path), 'Input file does not exist'
    output_file_path = args.input_file.replace('.json', f'_{args.partition}.json')
    with open(input_file_path, 'r') as f:
        data = json.load(f)
    new_data = {}
    for image_name, image_meta in tqdm(data.items()):
        cur_cam = image_meta['camera']
        cur_TMatrix = np.array(cur_cam[:16]).reshape(4, 4)
        cur_cam_scoord = get_cam_coords(cur_TMatrix)
        theta = cur_cam_scoord[0]
        if is_valid(theta, *partition):
            new_data[image_name] = image_meta
    with open(output_file_path, 'w') as f:
        json.dump(new_data, f, indent=4)

if __name__ == '__main__':  
    main()