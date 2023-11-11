'''
Author: tianhao 120090472@link.cuhk.edu.cn
Date: 2023-10-29 14:25:07
LastEditors: tianhao 120090472@link.cuhk.edu.cn
LastEditTime: 2023-10-29 16:41:31
FilePath: /DatProc/X7.remove.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import os, json, tqdm, shutil

input_path = '/data2/chence/PanoHeadData/single_view_hq/'
json_file_name = 'dataset.json'
data_sources = ["CelebA" ,"FFHQ", "K-Hairstyle/Training", "K-Hairstyle/Validation", "LPFF", "Web", "OCD/Original"]
needed_subdirs = ['align_images', 'align_masks', 'align_parsing']
output_path = '/data2/chence/PanoHeadData/single_view_hq_baseline'
json_path = os.path.join(input_path, json_file_name)
for subdir in needed_subdirs:
    os.makedirs(os.path.join(output_path, subdir), exist_ok=True)
with open(json_path, 'rb') as f:
    json_dict = json.load(f)     # key: '00000/img00000000.png' ...   
output_json = {}
for img_name, img_meta in tqdm.tqdm(json_dict.items()):
    img_source = img_meta['data_source']
    if img_source in data_sources:
        # add key, value to output_json
        output_json[img_name] = img_meta
        # copy image to output_path
        for subdir in needed_subdirs:
            _img_name = img_name.replace('png', 'jpg') if subdir != 'align_parsing' else img_name
            img_input_path = os.path.join(input_path, subdir, _img_name)
            img_output_path = os.path.join(output_path, subdir, _img_name)
            os.makedirs(os.path.dirname(img_output_path), exist_ok=True)
            shutil.copy(img_input_path, img_output_path)
            
# save output_json
with open(os.path.join(output_path, json_file_name), 'w') as f:
    json.dump(output_json, f, indent=4)
