import os
import shutil
from tqdm import tqdm

orgn_dataset_dir = '/datar/Web'
new_dataset_dir = '/datar/Web_small'
image_src_dir = os.path.join(orgn_dataset_dir, 'Data')
image_dst_dir = os.path.join(new_dataset_dir, 'Data')

os.makedirs(new_dataset_dir, exist_ok=True)
os.makedirs(image_dst_dir, exist_ok=True)

size_small_dataset = 1000

for i, image_name in tqdm(enumerate(os.listdir(image_src_dir)), total=size_small_dataset):
    if i >= size_small_dataset: break
    shutil.copy(os.path.join(image_src_dir, image_name), os.path.join(image_dst_dir, image_name))