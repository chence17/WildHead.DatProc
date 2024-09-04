import os
import os.path as osp
import tqdm
import json

def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def save_json(json_file, data):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

# split_tag = "Training"
split_tag = "Validation"

tgt_dir = f"data/K-Hairstyle/processed_data_0830/{split_tag}"
data_dir = f"{tgt_dir}/info_dict"
save_path = f"{tgt_dir}/dataset.json"
info_ext = ".json"
img_ext = ".jpg"

sub_save_dirs = ['image1024', 'image1024_face_mask', 'image1024_hair_mask', 'image1024_padding_mask', 'image563', 'image563_face_mask', 'image563_hair_mask', 'image563_padding_mask', 'info_dict']
sub_save_exts = ['.jpg', '.png', '.png', '.png', '.jpg', '.png', '.png', '.png', '.json']

full_info_dict = {}
sub_dirs = sorted(os.listdir(data_dir))
# print(f"==>> sub_dirs: {sub_dirs}")
for sub_dir in tqdm.tqdm(sub_dirs):
    sub_dir_path = osp.join(data_dir, sub_dir)
    sub_dir_files = sorted([f.replace(info_ext, '') for f in os.listdir(sub_dir_path) if f.endswith(info_ext)])
    # print(f"==>> sub_dir_files: {sub_dir_files}")
    for sub_dir_file in sub_dir_files:
        sub_dir_file_path = osp.join(sub_dir_path, sub_dir_file + info_ext)
        assert osp.exists(sub_dir_file_path), f"==>> {sub_dir_file_path} not exists"
        try:
            sub_dir_file_info = load_json(sub_dir_file_path)
        except:
            print(f"==>> {sub_dir_file_path} load failed")
            for ssd, sse in zip(sub_save_dirs, sub_save_exts):
                ssfile = osp.join(tgt_dir, ssd, sub_dir, f"{sub_dir_file}{sse}")
                if osp.exists(ssfile):
                    os.remove(ssfile)
                    print(f"==>> {ssfile} removed")
            continue
        full_info_dict[f"{sub_dir}/{sub_dir_file + img_ext}"] = sub_dir_file_info
save_json(save_path, full_info_dict)

