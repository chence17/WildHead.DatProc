'''
Author: tianhao 120090472@link.cuhk.edu.cn
Date: 2023-11-14 00:15:44
LastEditors: tianhao 120090472@link.cuhk.edu.cn
LastEditTime: 2024-02-25 21:35:33
FilePath: /DatProc/X11.build_testset_for_khs_remove.py
Description: 
    Remov unwanted items in K-Hairstyle
Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
# read dataset.json
import json, os, tqdm

ours_delete = []
khs_delete = []
khs_keep = []
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
            khs_delete.append((new_path, image_name))
        else:
            khs_keep.append((new_path, image_name))
    pbar.update()

print(f'In total: {len(khs_delete)} in K-Hairstyle will be deleted.')

out_dict = {
    'bad': {

    },
    'good': {

    }
}
test_model_nums = 100
for _path, image_name in khs_delete:
    model_id = os.path.basename(os.path.dirname(_path))
    if len(out_dict['bad']) >= test_model_nums:
        break
    if model_id in out_dict['bad'].keys():
        out_dict['bad'][model_id].append(image_name)
    else:
        out_dict['bad'][model_id] = [image_name]
for _path, image_name in khs_keep:
    model_id = os.path.basename(os.path.dirname(_path))
    if len(out_dict['good']) >= test_model_nums:
        break
    if model_id in out_dict['good'].keys():
        out_dict['good'][model_id].append(image_name)
    else:
        out_dict['good'][model_id] = [image_name]

    
outdir = '/home/shitianhao/project/DatProc/temp/test_model_ids.json'
with open(outdir, 'w') as f:
    json.dump(out_dict, f, indent=4)