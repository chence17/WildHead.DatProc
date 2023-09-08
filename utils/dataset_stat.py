import os
import cv2
import json
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Statistic')
    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--output_dir', type=str, help='path to output dir', default='/home/shitianhao/project/DatProc/utils/stats')
    args = parser.parse_args()
    return args

def get_all_extensions(data_path):
    extensions = set()
    for root, dirs, files in os.walk(data_path):
        for file in tqdm(files):
            ext = os.path.splitext(file)[-1]
            extensions.add(ext)
    return extensions

def stat_image_meta(data_path, target_file_exts=['.jpg', '.png']):
    img_meta = {}
    data_name = os.path.basename(data_path)
    for root, dirs, files in os.walk(data_path):
        for file in tqdm(files):
            ext = os.path.splitext(file)[-1]
            if ext in target_file_exts:
                photo_path = os.path.join(root, file)
                img = cv2.imread(photo_path)
                height, width, _ = img.shape
                photo_path = os.path.relpath(photo_path, data_path)
                img_meta[photo_path] = {'width': width, 'height': height}
    return img_meta

def main():
    args = parse_args()
    data_path = args.data_path
    assert os.path.exists(data_path), 'data path does not exist'
    data_name = os.path.basename(data_path)
    out_json_name = data_name + '_stat.json'
    out_json_path = os.path.join(args.output_dir, out_json_name)
    extensions = get_all_extensions(data_path)
    print(f'Found extensions: {extensions}')
    IMG_FORMATS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".svg", ".ico", ".exif", ".raw", ".heic", ".jfif", ".tga", ".pdf", ".eps", ".ai", ".psd"]
    extensions = list(set(extensions).intersection(set(IMG_FORMATS)))
    print(f'Found image extensions: {extensions}')
    img_meta = stat_image_meta(data_path, extensions)
    with open(out_json_path, 'w') as f:
        json.dump(img_meta, f, indent=4)

if __name__ == '__main__':
    main()
