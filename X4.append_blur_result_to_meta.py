import os
import json
import tqdm

blur_det_result_meta_path = '/data/PanoHeadData/single_view_hq/dataset_blur.json'
origin_meta_path = '/data/PanoHeadData/single_view_hq/dataset.json'


print(f'Loading {blur_det_result_meta_path}...')
with open(blur_det_result_meta_path, 'r') as f:
    blur_det_result_meta = json.load(f)
print(f'Loading {origin_meta_path}...')
with open(origin_meta_path, 'r') as f:
    origin_meta = json.load(f)

for img_key, img_val in tqdm.tqdm(blur_det_result_meta.items()):
    assert img_key in origin_meta.keys()
    origin_meta[img_key].update(img_val)

print(f'Saving to {origin_meta_path}...')
with open(origin_meta_path, 'w') as f:
    json.dump(origin_meta, f, indent=4)
