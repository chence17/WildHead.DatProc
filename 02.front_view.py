import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from utils.face_landmark import FaceAlignmentDetector
from utils.recrop_images import FrontViewRecropper
from utils.face_parsing import HeadParser
from utils.fv_utils import rotate_image, generate_results, crop_head_image, crop_head_parsing
from utils.tool import R2hpose


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

    flmk_det = FaceAlignmentDetector()
    recropper = FrontViewRecropper()
    hpar = HeadParser()
    head_image_size = 1024

    # inverse convert from OpenCV camera
    convert = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]).astype(np.float32)
    inv_convert = np.linalg.inv(convert)

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
        image_path = os.path.join(save_folder, dtitem['raw']['file_path'])
        image_name = os.path.basename(image_path)[:-4]
        image_data = cv2.imread(image_path)
        head_boxes = dtitem['raw']['head_boxes']
        dtdict[dtkey]['raw']['landmarks'] = {}
        dtdict[dtkey]['raw']['raw_quad'] = {}
        dtdict[dtkey]['raw']['tgt_quad'] = {}
        dtdict[dtkey]['raw']['rot_quad'] = {}
        dtdict[dtkey]['raw']['q2b_tf'] = {}
        dtdict[dtkey]['raw']['rotmat'] = {}
        dtdict[dtkey]['raw']['head_boxes_resize'] = head_image_size
        dtdict[dtkey]['head'] = {}
        for box_id, box in head_boxes.items():
            dtdict[dtkey]['head'][box_id] = {}
            try:
                box_np = np.array(box)
                head_image = crop_head_image(image_data.copy(), box_np)
                assert head_image.shape[0] == head_image.shape[1]
                landmarks = flmk_det(head_image, True, box_np[:2])
                assert landmarks is not None
                assert np.sum(landmarks < 0) == 0
                dtdict[dtkey]['raw']['landmarks'][box_id] = landmarks.tolist()

                cropped_img, camera_poses, quad, tf_quad = recropper(image_data.copy(), landmarks)
                quad, tf_quad = np.array(quad), np.array(tf_quad)
                dtdict[dtkey]['head'][box_id]['camera'] = camera_poses.tolist()
                dtdict[dtkey]['raw']['raw_quad'][box_id] = quad.tolist()
                dtdict[dtkey]['raw']['tgt_quad'][box_id] = tf_quad.tolist()

                # cam2world
                c2wR = camera_poses[:16].reshape(4, 4)[:3, :3]
                w2cR = inv_convert @ c2wR
                hpose = R2hpose(w2cR)
                dtdict[dtkey]['head'][box_id]['hpose'] = hpose  # yaw, roll, pitch

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

                rotated_image, rotmat, rot_quad = rotate_image(image_data.copy(), quad, tf_quad, borderMode=cv2.BORDER_REFLECT, upsample=2)
                dtdict[dtkey]['raw']['rotmat'][box_id] = rotmat.tolist()
                dtdict[dtkey]['raw']['rot_quad'][box_id] = rot_quad.tolist()

                head_image, head_crop_box, head_rot_quad = generate_results(rotated_image, rot_quad, box_np, head_image_size)
                dtdict[dtkey]['head'][box_id]['align_box'] = head_crop_box.tolist() # [x1, y1, w, h]
                dtdict[dtkey]['head'][box_id]['align_quad'] = head_rot_quad.tolist() # [tl, bl, br, tr]

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
                dtdict[dtkey]['head'][box_id]['view'] = 'front'
                dtdict[dtkey]['head'][box_id]['head_image_path'] = head_image_path
                dtdict[dtkey]['head'][box_id]['head_parsing_path'] = head_parsing_path
                dtdict[dtkey]['head'][box_id]['align_image_path'] = align_image_path
                dtdict[dtkey]['head'][box_id]['align_parsing_path'] = align_parsing_path
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
