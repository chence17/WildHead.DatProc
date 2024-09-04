import os
import os.path as osp
import tqdm

split_tag = "Training"
# split_tag = "Validation"

src_dir = f"data/K-Hairstyle/original_data/{split_tag}/rawset_crop/0003.rawset"
tgt_dir = f"data/K-Hairstyle/processed_data_0830/{split_tag}"

sub_save_dirs = ['image1024', 'image1024_face_mask', 'image1024_hair_mask', 'image1024_padding_mask', 'image563', 'image563_face_mask', 'image563_hair_mask', 'image563_padding_mask', 'info_dict']
sub_save_exts = ['.jpg', '.png', '.png', '.png', '.jpg', '.png', '.png', '.png', '.json']

for sub_save_dir in sub_save_dirs:
    os.makedirs(osp.join(tgt_dir, sub_save_dir), exist_ok=True)

src_sub_dirs = sorted(os.listdir(src_dir))
print(f"==>> src_sub_dirs: {src_sub_dirs}")
for src_sub_dir in src_sub_dirs:
    src_sub_dir_path = osp.join(src_dir, src_sub_dir)
    src_sub_sub_dirs = sorted(os.listdir(src_sub_dir_path))
    print(f"==>> src_sub_sub_dirs: {src_sub_sub_dirs}")
    for src_sub_sub_dir in tqdm.tqdm(src_sub_sub_dirs):
        src_sub_sub_dir_path = osp.join(src_sub_dir_path, src_sub_sub_dir)
        valid_image_files = None
        for ssd, sse in zip(sub_save_dirs, sub_save_exts):
            cur_filenames = set([f.replace(sse, '') for f in os.listdir(osp.join(src_sub_sub_dir_path, ssd)) if f.endswith(sse)])
            if valid_image_files is None:
                valid_image_files = cur_filenames
            else:
                valid_image_files = valid_image_files.intersection(cur_filenames)
        valid_image_files = sorted(list(valid_image_files))
        print(f"==>> valid_image_files: {valid_image_files}")
        for valid_image_file in valid_image_files:
            for ssd, sse in zip(sub_save_dirs, sub_save_exts):
                src_file = osp.join(src_sub_sub_dir_path, ssd, f"{valid_image_file}{sse}")
                assert osp.exists(src_file), f"src_file not exists: {src_file}"
                tgt_file = osp.join(tgt_dir, ssd, src_sub_sub_dir, f"{valid_image_file}{sse}")
                assert not osp.exists(tgt_file), f"tgt_file exists: {tgt_file}"
                os.makedirs(osp.dirname(tgt_file), exist_ok=True)
                os.system(f"cp {src_file} {tgt_file}")
