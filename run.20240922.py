import os
import os.path as osp
import json
import tqdm
import numpy as np
from dpmain.datproc_v3 import DatProcV3
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

    return partitioned_list, cur_start_idx


def get_all_images_in_folder(folder_path):
    # 定义图片的扩展名，可以根据需要添加更多的格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    image_files = []
    
    # 遍历文件夹及子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):  # 判断文件是否是图片
                image_files.append(os.path.join(root, file))  # 将完整路径添加到列表中
    
    return image_files


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=7 python run.20240812.py --data_source "KHS/Training" --img_root_dir /data/K-Hairstyle/Training/rawset/images/0003.rawset --json_root_dir /data/K-Hairstyle/Training/rawset/labels/0003.rawset --save_root_dir /data/K-Hairstyle/Training/rawset_crop/0003.rawset --total_splits 4 --split_index 0
    import argparse
    parser = argparse.ArgumentParser(
        description='Filter videos based on certain criteria.')
    parser.add_argument('--data_source', type=str,
                        required=True, help='Data source.')
    parser.add_argument("--img_root_dir", type=str,
                        required=True, help="Directory for the image root.")
    parser.add_argument("--save_root_dir", type=str,
                        required=True, help="Directory to save the output.")
    parser.add_argument('--total_splits', type=int,
                        required=True, help='Total number of splits.')
    parser.add_argument('--split_index', type=int,
                        required=True, help='Index of the split. 0, 1, 2')
    args = parser.parse_args()
    print(f"==>> args: {args}")

    dp = DatProcV3(args.data_source)

    img_root_dir = args.img_root_dir
    img_file_paths = get_all_images_in_folder(img_root_dir)
    img_file_paths = sorted(img_file_paths)

    save_root_dir = args.save_root_dir

    pos1_list, cur_start_index = partition_list(img_file_paths, args.split_index, args.total_splits)

    # info_dict_save_dir = osp.join(save_root_dir, "info_dict")
    # status_save_dir = osp.join(save_root_dir, "status")
    # head_image_save_dir = osp.join(save_root_dir, "image1024")
    # head_image_par_save_dir = osp.join(save_root_dir, "image1024_head_parsing")
    # head_image_msk_save_dir = osp.join(save_root_dir, "image1024_head_mask")
    # head_pad_mask_save_dir = osp.join(save_root_dir, "image1024_padding_mask")
    # cropped_img_save_dir = osp.join(save_root_dir, "image563")
    # cropped_img_par_save_dir = osp.join(save_root_dir, "image563_head_parsing")
    # cropped_img_msk_save_dir = osp.join(save_root_dir, "image563_head_mask")
    # cropped_pad_mask_save_dir = osp.join(save_root_dir, "image563_padding_mask")

    # for i in [info_dict_save_dir, status_save_dir, head_image_save_dir, head_image_par_save_dir, head_image_msk_save_dir, head_pad_mask_save_dir, cropped_img_save_dir, cropped_img_par_save_dir, cropped_img_msk_save_dir, cropped_pad_mask_save_dir]:
    #     os.makedirs(i, exist_ok=True)

    # for idx, pos1 in enumerate(pos1_list):
    #     try:
    #         img_name = osp.basename(pos1)
    #         img_name = img_name.split(".")[0]
    #         img_name = f"{cur_start_index:07d}_{img_name}"
    #         status_save_path = osp.join(status_save_dir, f"{img_name}.txt")

    #         if osp.exists(status_save_path):
    #             continue

    #         img_path = pos1
    #         assert osp.exists(img_path), f"Image file not found: {img_path}"

    #         proc_results = dp(img_path, use_landmarks=False)
    #         for procidx in range(len(proc_results)):
    #             proc_dict = proc_results[procidx]
    #             info_dict = proc_dict['info_dict']
    #             head_image = proc_dict['head_image']
    #             head_image_par = proc_dict['head_image_par']
    #             head_image_msk = proc_dict['head_image_msk']
    #             head_pad_mask = proc_dict['head_pad_mask']
    #             cropped_img = proc_dict['cropped_img']
    #             cropped_img_par = proc_dict['cropped_img_par']
    #             cropped_img_msk = proc_dict['cropped_img_msk']
    #             cropped_pad_mask = proc_dict['cropped_pad_mask']

    #             info_dict['raw_image_path'] = img_path

    #             head_image_save_path = osp.join(
    #                 head_image_save_dir, f"{img_name}_{procidx:02d}.jpg")
    #             head_image_par_save_path = osp.join(
    #                 head_image_par_save_dir, f"{img_name}_{procidx:02d}.png")
    #             head_image_msk_save_path = osp.join(
    #                 head_image_msk_save_dir, f"{img_name}_{procidx:02d}.png")
    #             head_pad_mask_save_path = osp.join(
    #                 head_pad_mask_save_dir, f"{img_name}_{procidx:02d}.png")
    #             cropped_img_save_path = osp.join(
    #                 cropped_img_save_dir, f"{img_name}_{procidx:02d}.jpg")
    #             cropped_img_par_save_path = osp.join(
    #                 cropped_img_par_save_dir, f"{img_name}_{procidx:02d}.png")
    #             cropped_img_msk_save_path = osp.join(
    #                 cropped_img_msk_save_dir, f"{img_name}_{procidx:02d}.png")
    #             cropped_pad_mask_save_path = osp.join(
    #                 cropped_pad_mask_save_dir, f"{img_name}_{procidx:02d}.png")

    #             with open(osp.join(info_dict_save_dir, f"{img_name}_{procidx:02d}.json"), "w") as f:
    #                 json.dump(info_dict, f)

    #             Image.fromarray(head_image).save(
    #                 head_image_save_path)
    #             Image.fromarray(head_image_par).save(
    #                 head_image_par_save_path)  # Face Mask
    #             Image.fromarray(head_image_msk).save(
    #                 head_image_msk_save_path)  # Hair Mask
    #             Image.fromarray(head_pad_mask).save(
    #                 head_pad_mask_save_path)  # Padding Mask
    #             Image.fromarray(cropped_img).save(
    #                 cropped_img_save_path)
    #             Image.fromarray(cropped_img_par).save(
    #                 cropped_img_par_save_path)  # Face Mask
    #             Image.fromarray(cropped_img_msk).save(
    #                 cropped_img_msk_save_path)  # Hair Mask
    #             Image.fromarray(cropped_pad_mask).save(
    #                 cropped_pad_mask_save_path)  # Padding Mask

    #         with open(status_save_path, "w") as f:
    #             f.write("Done")

    #     except Exception as e:
    #         print(f"Error: {idx} {pos1}")
    #         print(e)
    #         continue
