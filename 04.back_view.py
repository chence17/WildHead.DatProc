import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm

from utils.recrop_images import crop_final
from utils.face_parsing import HeadParser
from utils.fv_utils import generate_results, crop_head_parsing
from utils.head_pose_estimation import WHENetHeadPoseEstimator
from utils.bv_utils import estimate_rotation_angle, rotate_image, rotate_quad, get_final_crop_size
from utils.tool import transform_box, hpose2camera, box2quad


def parse_args():
    parser = argparse.ArgumentParser(description='Filter images.')
    parser.add_argument('-i', '--json_file', type=str, help='path to the json file', required=True)
    args, _ = parser.parse_known_args()

    if not os.path.isabs(args.json_file):
        args.json_file = os.path.abspath(args.json_file)
        print("Set json file to be absolute:", args.json_file)
    assert os.path.exists(args.json_file), f'args.json_file {args.json_file} does not exist!'

    return args


def main(args):
    print(args)
    # load json file
    with open(args.json_file, 'r', encoding='utf8') as f:
        dtdict = json.load(f)

    hpar = HeadParser()
    pe = WHENetHeadPoseEstimator('assets/whenet_1x3x224x224_prepost.onnx')
    assert pe.input_height == pe.input_width

    scale = [0.7417686609206039, 0.7417686609206039]
    shift = [-0.007425799169690871, 0.00886478197975557]
    head_image_size = 1024
    crop_size = get_final_crop_size(512)

    save_folder = os.path.dirname(args.json_file)
    head_image_folder = os.path.join(save_folder, 'head_images')
    os.makedirs(head_image_folder, exist_ok=True)
    head_parsing_folder = os.path.join(save_folder, 'head_parsing')
    os.makedirs(head_parsing_folder, exist_ok=True)
    align_image_folder = os.path.join(save_folder, 'align_images')
    os.makedirs(align_image_folder, exist_ok=True)
    align_parsing_folder = os.path.join(save_folder, 'align_parsing')
    os.makedirs(align_parsing_folder, exist_ok=True)
    print("head_image_folder:", head_image_folder)
    print("head_parsing_folder:", head_parsing_folder)
    print("align_image_folder:", align_image_folder)
    print("align_parsing_folder:", align_parsing_folder)

    for dtkey, dtitem in tqdm(dtdict.items()):
        head_boxes = dtitem['raw']['head_boxes']

        need_process = False
        for box_id, box in head_boxes.items():
            if box_id not in dtdict[dtkey]['head'].keys():
                need_process = True
                break
            if dtdict[dtkey]['head'][box_id]['view'] is None:
                need_process = True
                break
        if not need_process:
            continue

        image_path = os.path.join(save_folder, dtitem['raw']['file_path'])
        image_name = os.path.basename(image_path)[:-4]
        image_data = cv2.imread(image_path)
        for box_id, box in head_boxes.items():
            if box_id in dtdict[dtkey]['head'].keys():
                if dtdict[dtkey]['head'][box_id]['view'] is not None:
                    continue
            try:
                if box_id not in dtdict[dtkey]['raw']['landmarks'].keys():
                    dtdict[dtkey]['raw']['landmarks'][box_id] = None

                box_np = np.array(box)
                rot_quad = box2quad(transform_box(box, scale, shift))
                rot_angle, rot_center = estimate_rotation_angle(image_data.copy(), box_np, pe, iterations=3)
                rotated_image, rotmat = rotate_image(image_data.copy(), rot_center, rot_angle)
                dtdict[dtkey]['raw']['rotmat'][box_id] = rotmat.tolist()
                dtdict[dtkey]['raw']['rot_quad'][box_id] = rot_quad.tolist()

                quad = rotate_quad(rot_quad, rot_center, -rot_angle)
                cropped_img, _, tf_quad = crop_final(image_data.copy(), size=crop_size, quad=quad, top_expand=0.,
                                                     left_expand=0., bottom_expand=0., right_expand=0.)
                dtdict[dtkey]['raw']['raw_quad'][box_id] = quad.tolist()
                dtdict[dtkey]['raw']['tgt_quad'][box_id] = tf_quad.tolist()

                head_image, head_crop_box, head_rot_quad = generate_results(rotated_image, rot_quad, box_np,
                                                                            head_image_size)
                dtdict[dtkey]['head'][box_id]['align_box'] = head_crop_box.tolist()  # [x1, y1, w, h]
                dtdict[dtkey]['head'][box_id]['align_quad'] = head_rot_quad.tolist()  # [tl, bl, br, tr]

                hpose = pe(head_image, isBGR=True)
                dtdict[dtkey]['head'][box_id]['hpose'] = hpose.astype(np.float32).tolist()  # yaw, roll, pitch
                camera_poses = hpose2camera(hpose)
                dtdict[dtkey]['head'][box_id]['camera'] = camera_poses.tolist()

                quad_w = np.linalg.norm(quad[2] - quad[1])
                quad_h = np.linalg.norm(quad[1] - quad[0])
                quad_center = np.mean(quad, axis=0)
                hbox_w = box_np[2]
                hbox_h = box_np[3]
                hbox_center = box_np[:2] + box_np[2:] / 2.
                dtdict[dtkey]['raw']['q2b_tf'][box_id] = {
                    'scale': [quad_w / hbox_w, quad_h / hbox_h],
                    'shift': [(quad_center[0] - hbox_center[0]) / hbox_w, (quad_center[1] - hbox_center[1]) / hbox_h]
                }

                head_parsing = hpar(head_image, is_bgr=True, show=False)
                cropped_par = crop_head_parsing(head_parsing.copy(), head_crop_box)
                cropped_par = cv2.resize(cropped_par, (cropped_img.shape[1], cropped_img.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)

                head_image_path = os.path.join(head_image_folder, f"{image_name}_{box_id}.png")
                cv2.imwrite(head_image_path, head_image)
                head_parsing_path = os.path.join(head_parsing_folder, f"{image_name}_{box_id}.png")
                cv2.imwrite(head_parsing_path, head_parsing)

                align_image_path = os.path.join(align_image_folder, f"{image_name}_{box_id}.png")
                cv2.imwrite(align_image_path, cropped_img)
                align_parsing_path = os.path.join(align_parsing_folder, f"{image_name}_{box_id}.png")
                cv2.imwrite(align_parsing_path, cropped_par)
                dtdict[dtkey]['head'][box_id]['view'] = 'back'
                dtdict[dtkey]['head'][box_id]['head_image_path'] = os.path.relpath(head_image_path, save_folder)
                dtdict[dtkey]['head'][box_id]['head_parsing_path'] = os.path.relpath(head_parsing_path, save_folder)
                dtdict[dtkey]['head'][box_id]['align_image_path'] = os.path.relpath(align_image_path, save_folder)
                dtdict[dtkey]['head'][box_id]['align_parsing_path'] = os.path.relpath(align_parsing_path, save_folder)
            except:
                if box_id not in dtdict[dtkey]['raw']['landmarks'].keys():
                    dtdict[dtkey]['raw']['landmarks'][box_id] = None
                dtdict[dtkey]['head'][box_id]['camera'] = None
                dtdict[dtkey]['raw']['raw_quad'][box_id] = None
                dtdict[dtkey]['raw']['tgt_quad'][box_id] = None
                dtdict[dtkey]['head'][box_id]['hpose'] = None
                dtdict[dtkey]['raw']['q2b_tf'][box_id] = None
                dtdict[dtkey]['raw']['rotmat'][box_id] = None
                dtdict[dtkey]['raw']['rot_quad'][box_id] = None
                dtdict[dtkey]['head'][box_id]['align_box'] = None
                dtdict[dtkey]['head'][box_id]['align_quad'] = None
                dtdict[dtkey]['head'][box_id]['view'] = None
                dtdict[dtkey]['head'][box_id]['head_image_path'] = None
                dtdict[dtkey]['head'][box_id]['head_parsing_path'] = None
                dtdict[dtkey]['head'][box_id]['align_image_path'] = None
                dtdict[dtkey]['head'][box_id]['align_parsing_path'] = None

    with open(args.json_file, 'w', encoding='utf8') as f:
        json.dump(dtdict, f, indent=4)


if __name__ == '__main__':
    # Camera Checked.
    args = parse_args()
    main(args)

