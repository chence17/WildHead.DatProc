# 01 filter split
# 02 front view
# 02.5 head ratio stat
# 03 back view
import os
import cv2
import numpy as np
import json
import tqdm

from utils.filter import filter_invalid_and_small
from utils.head_detection import YoloHeadDetector, resize_and_pad_image
from utils.face_landmark import FaceAlignmentDetector
from utils.face_parsing import HeadParser
from utils.recrop_images import Recropper
from back_view import WHENetHeadPoseEstimator, get_rotate_angle, hbox2quad, rotate_quad, crop_final, calculate_R
from back_view import eg3dcamparams

image_dir = 'temp/KHairstyle2/1800.DSS532777/images'
base_dir = os.path.dirname(image_dir)
head_image_dir = os.path.join(base_dir, 'head_images')
os.makedirs(head_image_dir, exist_ok=True)
align_image_dir = os.path.join(base_dir, 'align_images')
os.makedirs(align_image_dir, exist_ok=True)
align_semantic_dir = os.path.join(base_dir, 'align_semantic')
os.makedirs(align_semantic_dir, exist_ok=True)

hbox_det = YoloHeadDetector(weights_file='assets/224x224_yolov4_hddet_480x640.onnx',
                            input_width=640, input_height=480)
head_image_hw = (1024, 1024)
head_image_ext = '.jpg'
flmk_det = FaceAlignmentDetector()
recropper = Recropper()
hpar = HeadParser()
pe = WHENetHeadPoseEstimator('assets/whenet_1x3x224x224_prepost.onnx')
assert pe.input_height == pe.input_width

shrink_ratio = 0.75
crop_size = 563

labels = {}

for image_name in tqdm.tqdm(os.listdir(image_dir)):
    image_path = filter_invalid_and_small(os.path.join(image_dir, image_name))
    if image_path is None:
        print(f'{image_name} is invalid or too small')
        continue

    image_data = cv2.imread(image_path)
    head_boxes = hbox_det(image_data.copy(), isBGR=True)

    if head_boxes is None or head_boxes.shape[0] == 0:
        print(f'{image_name} no head')
        continue

    print(image_name, head_boxes)
    for idx, head_box in enumerate(head_boxes):
        head_image_name = f'{image_name[:-4]}_{idx}{head_image_ext}'
        head_semantic_name = f'{image_name[:-4]}_{idx}.png'
        x1, y1, w, h = head_box
        x2, y2 = x1 + w, y1 + h
        head_image = image_data[int(y1):int(y2), int(x1):int(x2)].copy()
        head_image = resize_and_pad_image(head_image, head_image_hw, border_type=cv2.BORDER_REFLECT)[0]
        cv2.imwrite(os.path.join(head_image_dir, head_image_name), head_image)
        landmarks = flmk_det(head_image, True)

        front_view_flag = True

        if landmarks is None:
            print(f'{image_name} {idx} landmarks is None')
            front_view_flag = False
        else:
            try:
                cropped_img, camera_poses, quad = recropper(head_image, landmarks)
                sem = hpar(cropped_img, is_bgr=True)
                cv2.imwrite(os.path.join(align_image_dir, head_image_name), cropped_img)
                cv2.imwrite(os.path.join(align_semantic_dir, head_semantic_name), sem)
                labels[head_image_name] = camera_poses
            except:
                front_view_flag = False

        if not front_view_flag:
            try:
                hbox = [0, 0, head_image.shape[1], head_image.shape[0]]
                angle, _ = get_rotate_angle(hbox, head_image, True, pe, iterations=3)
                x1, y1, hbox_w, hbox_h = hbox
                hbox_cx, hbox_cy = x1 + hbox_w / 2, y1 + hbox_h / 2
                hbox_w, hbox_h = hbox_w * shrink_ratio, hbox_h * shrink_ratio
                x1, y1 = hbox_cx - hbox_w / 2, hbox_cy - hbox_h / 2
                hbox = [x1, y1, hbox_w, hbox_h]
                hbox_quad = hbox2quad(hbox)
                hbox_center = np.mean(hbox_quad, axis=0)
                hbox_quad = rotate_quad(hbox_quad.copy(), angle, hbox_center)
                hbox_img = crop_final(head_image, size=crop_size, quad=hbox_quad,
                                      top_expand=0., left_expand=0., bottom_expand=0., right_expand=0.)
                hbox_img_sem = hpar(hbox_img, True)
                hbox_img_hpose = pe(cv2.resize(hbox_img, pe.input_hw), isBGR=True)
                R = calculate_R(hbox_img_hpose)
                s = 1
                t3d = np.array([0., 0., 0.])
                R[:, :3] = R[:, :3] * s
                P = np.concatenate([R, t3d[:, None]], 1)
                P = np.concatenate([P, np.array([[0, 0, 0, 1.]])], 0)
                cam = eg3dcamparams(P.flatten())
                cv2.imwrite(os.path.join(align_image_dir, head_image_name), hbox_img)
                cv2.imwrite(os.path.join(align_semantic_dir, head_semantic_name), hbox_img_sem)
                labels[head_image_name] = cam
            except:
                print(f'{image_name} {idx} back view failed')
                continue

results_new = []
for img, P in labels.items():
    # img = os.path.basename(img)
    res = [format(r, '.6f') for r in P]
    results_new.append((img, res))
with open(os.path.join(base_dir, 'dataset.json'), 'w') as outfile:
    json.dump({"labels": results_new}, outfile, indent="\t")
