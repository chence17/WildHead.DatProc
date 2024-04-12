from dpmain.datproc_v1 import DatProcV1
import argparse
import json
import os
import os.path as osp
import tqdm
import imagesize
from skimage import io
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Process data with start and end indices.')
    parser.add_argument('--start_index', type=int, help='start index')
    parser.add_argument('--end_index', type=int, help='end index')
    parser.add_argument('--process_dict', type=str, help='path to JSON file')
    return parser.parse_args()


# Parse command line arguments
args = parse_args()

# Get start and end indices from command line arguments
start_index = args.start_index
end_index = args.end_index

# Load dictionary from JSON file
assert osp.exists(args.process_dict), f"File {args.process_dict} does not exist."
with open(args.process_dict, 'r') as file:
    process_dict = json.load(file)

all_pkeys = sorted(process_dict.keys())
pro_pkeys = all_pkeys[start_index:end_index]

# Use the loaded dictionary
# Example: print the loaded data
print(f"All pkeys: {len(all_pkeys)}")
print(f"Start index: {start_index}, Start pkey: {pro_pkeys[0]}")
print(f"End index: {end_index}, End pkey: {pro_pkeys[-1]}")
print(f"Process pkeys: {len(pro_pkeys)}")

dp = DatProcV1("Web20240228/data_common")

view_dict = {}
view_file = osp.join(f'view_{start_index:08d}to{end_index:08d}.json')
cur_num = 0
for k in tqdm.tqdm(pro_pkeys):
    img_path = process_dict[k]
    view = None
    try:
        # Load image data
        img_data = io.imread(img_path)  # RGB uint8 HW3 ndarray
        # Detect head box
        hed_boxes = dp.hed_det(img_data, isBGR=False, max_box_num=1) # N * 4 np.ndarray, each box is [x_min, y_min, w, h].
        box_np = np.array(hed_boxes[0])  # Coords in Raw Image
        head_image = dp.crop_head_image(img_data.copy(), box_np)
        # Estimate head pose and camera poses
        hpose = dp.hed_pe(head_image, isBGR=False)
        cur_cam = dp.hpose2camera(hpose).tolist()
        cur_TMatrix = np.array(cur_cam[:16]).reshape(4, 4)
        cur_cam_scoord = dp.get_cam_coords(cur_TMatrix)  # [theta, phi, r, x, y, z]
        # front [45, 135]
        # right [-45, 45]
        # back [-135, -45]
        # left [-180, -135], [135, 180]
        theta = cur_cam_scoord[0]
        if theta >= -45 and theta <= 45:
            view = 'right'
        elif theta >= 45 and theta <= 135:
            view = 'front'
        elif theta >= -135 and theta <= -45:
            view = 'back'
        else:
            view = 'left'
    except Exception as e:
        # print(f"Error: {e}")
        pass
    view_dict[k] = view
    cur_num += 1
    if cur_num % 10000 == 0:
        print(f"Processed {cur_num} images. Saving meta data.")
        with open(view_file, 'w') as file:
            json.dump(view_dict, file, indent=4)

with open(view_file, 'w') as file:
    json.dump(view_dict, file, indent=4)
