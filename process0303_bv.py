'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-02-28 23:03:57
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2024-03-01 09:41:54
FilePath: /DatProc/process0228.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from dpmain.datproc_v1 import DatProcV1
import argparse
import json
import os
import os.path as osp
import tqdm
import imagesize
from PIL import Image


def calc_iv2b_ratio(cur_box, raw_image_path):
    bx_min, by_min, bw, bh = cur_box
    bx_max, by_max = bx_min + bw, by_min + bh
    img_w, img_h = imagesize.get(raw_image_path)
    vx_min, vy_min = max(0, bx_min), max(0, by_min)
    vx_max, vy_max = min(img_w, bx_max), min(img_h, by_max)
    vw, vh = vx_max - vx_min, vy_max - vy_min
    v2b_ratio = (vw * vh) / (bw * bh)
    iv2b_ratio = 1.0 - v2b_ratio
    return iv2b_ratio


def parse_args():
    parser = argparse.ArgumentParser(description='Process data with start and end indices.')
    parser.add_argument('--start_index', type=int, help='start index')
    parser.add_argument('--end_index', type=int, help='end index')
    parser.add_argument('--process_dict', type=str, help='path to JSON file')
    parser.add_argument('--output_dir', type=str, help='output directory')
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

dp = DatProcV1("Web20240303/data_rare_bv")
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

info_meta = {}
formated_info_meta = {}

info_meta_file = osp.join(output_dir, f'info_meta_{start_index:08d}to{end_index:08d}.json')
formated_info_meta_file = osp.join(output_dir, f'meta_{start_index:08d}to{end_index:08d}.json')

if osp.exists(info_meta_file):
    with open(info_meta_file, 'r') as file:
        info_meta = json.load(file)

if osp.exists(formated_info_meta_file):
    with open(formated_info_meta_file, 'r') as file:
        formated_info_meta = json.load(file)

cur_num = 0
for pkey in tqdm.tqdm(pro_pkeys):
    try:
        meta_pkey = f'{pkey}.jpg'
        if meta_pkey in info_meta and meta_pkey in formated_info_meta:
            continue
        image_path = process_dict[pkey]
        folder_name, image_prefix = pkey.split('/')

        align_images_folder = os.path.join(os.path.join(output_dir, 'align_images'), folder_name)
        align_parsing_folder = os.path.join(os.path.join(output_dir, 'align_parsing'), folder_name)
        align_masks_folder = os.path.join(os.path.join(output_dir, 'align_masks'), folder_name)
        head_images_folder = os.path.join(os.path.join(output_dir, 'head_images'), folder_name)
        head_parsing_folder = os.path.join(os.path.join(output_dir, 'head_parsing'), folder_name)
        head_masks_folder = os.path.join(os.path.join(output_dir, 'head_masks'), folder_name)
        os.makedirs(align_images_folder, exist_ok=True)
        os.makedirs(align_parsing_folder, exist_ok=True)
        os.makedirs(align_masks_folder, exist_ok=True)
        os.makedirs(head_images_folder, exist_ok=True)
        os.makedirs(head_parsing_folder, exist_ok=True)
        os.makedirs(head_masks_folder, exist_ok=True)

        align_image_path = os.path.join(align_images_folder, f'{image_prefix}.jpg')
        align_parsing_path = os.path.join(align_parsing_folder, f'{image_prefix}.png')
        align_mask_path = os.path.join(align_masks_folder, f'{image_prefix}.png')
        head_image_path = os.path.join(head_images_folder, f'{image_prefix}.jpg')
        head_parsing_path = os.path.join(head_parsing_folder, f'{image_prefix}.png')
        head_mask_path = os.path.join(head_masks_folder, f'{image_prefix}.png')

        info_dict, head_image, head_image_par, head_image_msk, cropped_img, cropped_img_par, cropped_img_msk = dp(image_path)

        formated_info_dict = {
            'data_source': info_dict['data_source'],
            'camera': info_dict['head']['camera'],
            'hpose': info_dict['head']['hpose'],
            'align_box': info_dict['head']['align_box'],
            'align_quad': info_dict['head']['align_quad'],
            'view': info_dict['head']['view'],
            'camera_scoord': info_dict['head']['camera_scoord'],
            'align_image_path': osp.relpath(align_image_path, output_dir),
            'align_parsing_path': osp.relpath(align_parsing_path, output_dir),
            'align_mask_path': osp.relpath(align_mask_path, output_dir),
            'head_image_path': osp.relpath(head_image_path, output_dir),
            'head_parsing_path': osp.relpath(head_parsing_path, output_dir),
            'head_mask_path': osp.relpath(head_mask_path, output_dir),
            'svd_score': info_dict['head']['svd_score'],
            'laplacian_score': info_dict['head']['laplacian_score'],
            'iv2b_ratio': calc_iv2b_ratio(info_dict['raw']['box'], image_path),
            'head_region_thresh': info_dict['head']['par_ratio'],
            'dup_num': 1,
            'raw_image_path': osp.relpath(image_path, '/home/ce.chen/chence/Data/Datasets/Head'),
            'head_region_parsing_ratio': info_dict['head']['par_ratio'],
            'head_region_mask_ratio': info_dict['head']['msk_ratio']
        }

        Image.fromarray(head_image).save(head_image_path)
        Image.fromarray(head_image_par).save(head_parsing_path)
        Image.fromarray(head_image_msk).save(head_mask_path)
        Image.fromarray(cropped_img).save(align_image_path)
        Image.fromarray(cropped_img_par).save(align_parsing_path)
        Image.fromarray(cropped_img_msk).save(align_mask_path)
        info_meta[meta_pkey] = info_dict
        formated_info_meta[meta_pkey] = formated_info_dict
    except Exception as e:
        print(f"Error: {pkey}, {e}")
    cur_num += 1
    if cur_num % 1000 == 0:
        print(f"Processed {cur_num} images. Saving meta data.")
        with open(info_meta_file, 'w') as file:
            json.dump(info_meta, file, indent=4)

        with open(formated_info_meta_file, 'w') as file:
            json.dump(formated_info_meta, file, indent=4)

with open(info_meta_file, 'w') as file:
    json.dump(info_meta, file, indent=4)

with open(formated_info_meta_file, 'w') as file:
    json.dump(formated_info_meta, file, indent=4)
