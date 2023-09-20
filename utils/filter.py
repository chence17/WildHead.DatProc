'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-09-18 12:55:49
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-09-18 21:46:30
FilePath: /DatProc/utils/filter.py
Description: filter
'''
import os
import imagesize


def is_small(w, h, min_size=512):
    return min(w, h) < min_size


def filter_invalid_and_small(img_path):
    try:
        if not os.path.exists(img_path):
            # not exist
            return None
        img_w, img_h = imagesize.get(img_path)
        if is_small(img_h, img_w):
            # small images
            return None
        else:
            # valid images
            return img_path
    except:
        # invalid images
        return None


def load_image_names(image_folder):
    image_paths = []
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    for file_name in os.listdir(image_folder):
        ext = os.path.splitext(file_name)[-1]
        if ext in image_extensions:
            image_paths.append(os.path.join(image_folder, file_name))
    return image_paths