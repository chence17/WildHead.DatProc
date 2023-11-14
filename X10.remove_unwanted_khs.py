'''
Author: tianhao 120090472@link.cuhk.edu.cn
Date: 2023-11-14 00:15:44
LastEditors: tianhao 120090472@link.cuhk.edu.cn
LastEditTime: 2023-11-14 14:16:27
FilePath: /DatProc/X10.remove_unwanted_khs.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
# read dataset.json
import json, os, tqdm

ours_delete = []
khs_delete = []
label_root_dir = '/data3/khs_labels/'
dataset_json_path = '/data2/chence/PanoHeadData/multi_view_hq/dataset.json'
suspected_samples = [
    "묶음머리",
    "넘긴머리",
    "악성곱슬머리",
    "땋은머리",
    "기타악세사리",
    "기타"
]

print(f'Loading present dataset meta from: {dataset_json_path}', end='...', flush=True)
with open(dataset_json_path, 'r') as f:
    dataset = json.load(f)
print(f'Done.')

# get datasource
print(f'Filtering KHS data...')
pbar = tqdm.tqdm(dataset.items())
for image_name, image_meta in pbar:
    source = image_meta['data_source']
    if source == 'OCD/Original':
        pbar.update()
        ours_delete.append(image_name)
        continue
    if not(source.startswith('K-Hairstyle')) or source.endswith('-VF'): 
        pbar.update()
        continue
    path_label = 'validation_labels' if source == 'K-Hairstyle/Validation' else 'training_labels'
    new_path = image_meta['align_image_path'].replace('align_images', path_label)
    new_path = new_path.replace('._00.png', '.json') if new_path.endswith('._00.png') else new_path.replace('_00.png', '.json')
    new_path = new_path.replace('-', '_')
    abs_path = os.path.join(label_root_dir, new_path)
    if not(os.path.exists(abs_path)):
        print(f'Warning: {abs_path} does not exist.')
        pbar.update()
        continue
    pbar.set_description(f'Processing {new_path}')
    with open(abs_path, 'r') as f:
        label = json.load(f)
        exceptional = label['exceptional']
        if exceptional in suspected_samples:
            khs_delete.append(new_path)
    pbar.update()

print(f'In total: {len(khs_delete)} in K-Hairstyle will be deleted.')
print(f'In total: {len(ours_delete)} in OCD will be deleted.')

model_ids = set()
for _path in khs_delete:
    model_id = os.path.basename(os.path.dirname(_path))
    model_ids.add(model_id) 
print(f'In total: {len(model_ids)} models will be deleted.')