import os
import os.path as osp
import json
import tqdm
import numpy as np
from dpmain.datproc_v2 import DatProcV2
from PIL import Image


def partition_list(full_list, split_index, total_splits):
    full_list = sorted(full_list)

    assert split_index < total_splits, "split_index should be less than total_splits."

    num_per_split = int(np.ceil(len(full_list) / total_splits))
    cur_start_idx = split_index * num_per_split
    cur_end_idx = (split_index + 1) * num_per_split

    partitioned_list = full_list[cur_start_idx:cur_end_idx]

    print(f"cur_start_idx: {cur_start_idx}, cur_end_idx: {cur_end_idx}")
    print(
        f"full_list[0]: {partitioned_list[0]}, full_list[-1]: {partitioned_list[-1]}")

    return partitioned_list


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=7 python run.20240812.py --data_source "KHS/Training" --img_root_dir /data/K-Hairstyle/Training/rawset/images/0003.rawset --json_root_dir /data/K-Hairstyle/Training/rawset/labels/0003.rawset --save_root_dir /data/K-Hairstyle/Training/rawset_crop/0003.rawset --total_splits 4 --split_index 0
    import argparse
    parser = argparse.ArgumentParser(
        description='Filter videos based on certain criteria.')
    parser.add_argument('--data_source', type=str,
                        required=True, help='Data source.')
    parser.add_argument("--img_root_dir", type=str,
                        required=True, help="Directory for the image root.")
    parser.add_argument("--json_root_dir", type=str,
                        required=True, help="Directory for the JSON root.")
    parser.add_argument("--save_root_dir", type=str,
                        required=True, help="Directory to save the output.")
    parser.add_argument('--total_splits', type=int,
                        required=True, help='Total number of splits.')
    parser.add_argument('--split_index', type=int,
                        required=True, help='Index of the split. 0, 1, 2')
    args = parser.parse_args()
    print(f"==>> args: {args}")

    dp = DatProcV2(args.data_source)

    img_root_dir = args.img_root_dir
    json_root_dir = args.json_root_dir
    save_root_dir = args.save_root_dir

    pos1_list = partition_list(os.listdir(
        img_root_dir), args.split_index, args.total_splits)

    for pos1 in pos1_list:
        try:
            pos2_list = sorted(os.listdir(osp.join(img_root_dir, pos1)))
        except Exception as e:
            print(f"Error: {pos1}")
            print(e)
            continue

        for pos2 in pos2_list:
            try:
                img_dir = osp.join(img_root_dir, pos1, pos2)
                print(f"==>> img_dir: {img_dir}")
                json_dir = osp.join(json_root_dir, pos1, pos2)
                print(f"==>> json_dir: {json_dir}")
                save_dir = osp.join(save_root_dir, pos1, pos2)
                print(f"==>> save_dir: {save_dir}")
                info_dict_save_dir = osp.join(save_dir, "info_dict")
                head_image_save_dir = osp.join(save_dir, "image1024")
                head_image_par_save_dir = osp.join(
                    save_dir, "image1024_face_mask")
                head_image_msk_save_dir = osp.join(
                    save_dir, "image1024_hair_mask")
                head_pad_mask_save_dir = osp.join(
                    save_dir, "image1024_padding_mask")
                cropped_img_save_dir = osp.join(save_dir, "image563")
                cropped_img_par_save_dir = osp.join(
                    save_dir, "image563_face_mask")
                cropped_img_msk_save_dir = osp.join(
                    save_dir, "image563_hair_mask")
                cropped_pad_mask_save_dir = osp.join(
                    save_dir, "image563_padding_mask")

                for i in [info_dict_save_dir, head_image_save_dir, head_image_par_save_dir, head_image_msk_save_dir, head_pad_mask_save_dir, cropped_img_save_dir, cropped_img_par_save_dir, cropped_img_msk_save_dir, cropped_pad_mask_save_dir]:
                    os.makedirs(i, exist_ok=True)

                assert osp.exists(
                    img_dir), f"Image directory not found: {img_dir}"
                assert osp.exists(
                    json_dir), f"JSON directory not found: {json_dir}"

                img_files = sorted(os.listdir(img_dir))
            except Exception as e:
                print(f"Error: {pos2}")
                print(e)
                continue

            for img_file in tqdm.tqdm(img_files):
                try:
                    img_name = img_file.split(".")[0]
                    head_image_save_path = osp.join(
                        head_image_save_dir, f"{img_name}.jpg")
                    head_image_par_save_path = osp.join(
                        head_image_par_save_dir, f"{img_name}.png")
                    head_image_msk_save_path = osp.join(
                        head_image_msk_save_dir, f"{img_name}.png")
                    head_pad_mask_save_path = osp.join(
                        head_pad_mask_save_dir, f"{img_name}.png")
                    cropped_img_save_path = osp.join(
                        cropped_img_save_dir, f"{img_name}.jpg")
                    cropped_img_par_save_path = osp.join(
                        cropped_img_par_save_dir, f"{img_name}.png")
                    cropped_img_msk_save_path = osp.join(
                        cropped_img_msk_save_dir, f"{img_name}.png")
                    cropped_pad_mask_save_path = osp.join(
                        cropped_pad_mask_save_dir, f"{img_name}.png")

                    flag = True
                    flag = flag and osp.exists(head_image_save_path)
                    flag = flag and osp.exists(head_image_par_save_path)
                    flag = flag and osp.exists(head_image_msk_save_path)
                    flag = flag and osp.exists(head_pad_mask_save_path)
                    flag = flag and osp.exists(cropped_img_save_path)
                    flag = flag and osp.exists(cropped_img_par_save_path)
                    flag = flag and osp.exists(cropped_img_msk_save_path)
                    flag = flag and osp.exists(cropped_pad_mask_save_path)

                    if flag:
                        continue

                    json_file = img_file.replace(
                        ".jpg", ".json").replace("-", "_")

                    img_path = osp.join(img_dir, img_file)
                    json_path = osp.join(json_dir, json_file)

                    assert osp.exists(
                        img_path), f"Image file not found: {img_path}"
                    assert osp.exists(
                        json_path), f"JSON file not found: {json_path}"

                    info_dict, head_image, head_image_par, head_image_msk, head_pad_mask, cropped_img, cropped_img_par, cropped_img_msk, cropped_pad_mask = dp(
                        img_path, json_path, use_landmarks=False)
                    info_dict['raw_image_path'] = img_path
                    info_dict['raw_label_path'] = json_path

                    with open(osp.join(info_dict_save_dir, f"{img_name}.json"), "w") as f:
                        json.dump(info_dict, f)

                    Image.fromarray(head_image).save(
                        head_image_save_path)
                    Image.fromarray(head_image_par).save(
                        head_image_par_save_path)  # Face Mask
                    Image.fromarray(head_image_msk).save(
                        head_image_msk_save_path)  # Hair Mask
                    Image.fromarray(head_pad_mask).save(
                        head_pad_mask_save_path)  # Padding Mask
                    Image.fromarray(cropped_img).save(
                        cropped_img_save_path)
                    Image.fromarray(cropped_img_par).save(
                        cropped_img_par_save_path)  # Face Mask
                    Image.fromarray(cropped_img_msk).save(
                        cropped_img_msk_save_path)  # Hair Mask
                    Image.fromarray(cropped_pad_mask).save(
                        cropped_pad_mask_save_path)  # Padding Mask
                except Exception as e:
                    print(f"Error: {img_file}")
                    print(e)
                    continue
