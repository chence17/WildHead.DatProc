import os
import os.path as osp
import sys
import argparse
import tqdm

import torch
from PIL import Image
from torchvision.transforms import ToPILImage

from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms, utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_root', type=str, help='ocd dataset root path',
                        default='/data5/chence/PanoHeadData/single_view_hq/align_images')
    parser.add_argument('--batch_size', type=int, help='ocd dataset root path', default=16)
    parser.add_argument('--start_idx', type=int, required=True, help='start_idx')
    parser.add_argument('--end_idx', type=int, required=True, help='end_idx')
    args = parser.parse_args()
    assert osp.exists(args.ocd_root)
    args.ocd_root = osp.abspath(args.ocd_root)
    return args


if __name__ == "__main__":
    args = parse_args()
    data_root = osp.pardir(args.seg_root)
    print('data_root: ', data_root)
    mask_save_dir = osp.join(data_root, 'mask')
    image_folders = sorted([i for i in os.listdir(args.seg_root) if osp.isdir(osp.join(args.seg_root, i))])
    print('len(image_folders): ', len(image_folders))
    process_folders = image_folders[args.start_idx:args.end_idx]
    print('len(process_folders): ', len(process_folders))
    print('process_folders[0]: ', process_folders[0])
    print('process_folders[-1]: ', process_folders[-1])
    for i in tqdm.tqdm(process_folders):
        cur_image_folder = osp.join(args.seg_root, i)
        cur_image_files = sorted([j for j in os.listdir(cur_image_folder) if osp.isfile(osp.join(cur_image_folder, j))])
        print('len(cur_image_files): ', len(cur_image_files))
        print(cur_image_files)
    # Seg Mask
