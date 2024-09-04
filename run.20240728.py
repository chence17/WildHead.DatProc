import os
import os.path as osp
import json
import tqdm
from dpmain.datproc_v2 import DatProcV2
from PIL import Image

dp = DatProcV2("KHS/debug")

img_root_dir = "data/20240728.debug/rawset/images/0003.rawset"
json_root_dir = "data/20240728.debug/rawset/labels/0003.rawset"
save_root_dir = "output/20240728.debug/rawset_crop/0003.rawset"

pos1_list = sorted(os.listdir(img_root_dir))

for pos1 in pos1_list:
    pos2_list = sorted(os.listdir(osp.join(img_root_dir, pos1)))
    for pos2 in pos2_list:
        img_dir = osp.join(img_root_dir, pos1, pos2)
        print(f"==>> img_dir: {img_dir}")
        json_dir = osp.join(json_root_dir, pos1, pos2)
        print(f"==>> json_dir: {json_dir}")
        save_dir = osp.join(save_root_dir, pos1, pos2)
        print(f"==>> save_dir: {save_dir}")
        info_dict_save_dir = osp.join(save_dir, "info_dict")
        head_image_save_dir = osp.join(save_dir, "image1024")
        head_image_par_save_dir = osp.join(save_dir, "image1024_face_mask")
        head_image_msk_save_dir = osp.join(save_dir, "image1024_hair_mask")
        head_pad_mask_save_dir = osp.join(save_dir, "image1024_padding_mask")
        cropped_img_save_dir = osp.join(save_dir, "image563")
        cropped_img_par_save_dir = osp.join(save_dir, "image563_face_mask")
        cropped_img_msk_save_dir = osp.join(save_dir, "image563_hair_mask")
        cropped_pad_mask_save_dir = osp.join(save_dir, "image563_padding_mask")

        for i in [info_dict_save_dir, head_image_save_dir, head_image_par_save_dir, head_image_msk_save_dir, head_pad_mask_save_dir, cropped_img_save_dir, cropped_img_par_save_dir, cropped_img_msk_save_dir, cropped_pad_mask_save_dir]:
            os.makedirs(i, exist_ok=True)


        assert osp.exists(img_dir), f"Image directory not found: {img_dir}"
        assert osp.exists(json_dir), f"JSON directory not found: {json_dir}"

        img_files = sorted(os.listdir(img_dir))

        for img_file in tqdm.tqdm(img_files):
            try:
                json_file = img_file.replace(".jpg", ".json").replace("-", "_")

                img_path = osp.join(img_dir, img_file)
                json_path = osp.join(json_dir, json_file)

                assert osp.exists(img_path), f"Image file not found: {img_path}"
                assert osp.exists(json_path), f"JSON file not found: {json_path}"

                info_dict, head_image, head_image_par, head_image_msk, head_pad_mask, cropped_img, cropped_img_par, cropped_img_msk, cropped_pad_mask = dp(img_path, json_path, use_landmarks=False)
                info_dict['raw_image_path'] = img_path
                info_dict['raw_label_path'] = json_path
            except Exception as e:
                print(f"Error: {img_file}")
                print(e)
                continue

            img_name = img_file.split(".")[0]
            with open(osp.join(info_dict_save_dir, f"{img_name}.json"), "w") as f:
                json.dump(info_dict, f)

            Image.fromarray(head_image).save(osp.join(head_image_save_dir, f"{img_name}.jpg"))
            Image.fromarray(head_image_par).save(osp.join(head_image_par_save_dir, f"{img_name}.png"))  # Face Mask
            Image.fromarray(head_image_msk).save(osp.join(head_image_msk_save_dir, f"{img_name}.png"))  # Hair Mask
            Image.fromarray(head_pad_mask).save(osp.join(head_pad_mask_save_dir, f"{img_name}.png"))  # Padding Mask
            Image.fromarray(cropped_img).save(osp.join(cropped_img_save_dir, f"{img_name}.jpg"))
            Image.fromarray(cropped_img_par).save(osp.join(cropped_img_par_save_dir, f"{img_name}.png"))  # Face Mask
            Image.fromarray(cropped_img_msk).save(osp.join(cropped_img_msk_save_dir, f"{img_name}.png"))  # Hair Mask
            Image.fromarray(cropped_pad_mask).save(osp.join(cropped_pad_mask_save_dir, f"{img_name}.png"))  # Padding Mask
