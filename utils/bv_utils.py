'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-09-20 18:27:29
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-10-04 20:34:23
FilePath: /DatProc/utils/bv_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np
from utils.fv_utils import crop_head_image
from utils.tool import calculate_y_angle, hpose2R

def rotate_image(img, center, angle, borderMode=cv2.BORDER_REFLECT, upsample=2):
    rotmat = cv2.getRotationMatrix2D(center, angle, 1)
    crop_w = img.shape[1]
    crop_h = img.shape[0]
    crop_size = (crop_w, crop_h)
    if upsample is None or upsample == 1:
        crop_img = cv2.warpAffine(np.array(img), rotmat, crop_size, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
    else:
        assert isinstance(upsample, int)
        crop_size_large = (crop_w * upsample, crop_h * upsample)
        crop_img = cv2.warpAffine(np.array(img), upsample * rotmat, crop_size_large, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
        crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_AREA)

    empty = np.ones_like(img) * 255
    crop_mask = cv2.warpAffine(empty, rotmat, crop_size)

    if True:
        size = min(crop_w, crop_h)
        mask_kernel = int(size*0.02)*2+1
        blur_kernel = int(size*0.03)*2+1

        if crop_mask.mean() < 255:
            blur_mask = cv2.blur(crop_mask.astype(np.float32).mean(2),(mask_kernel,mask_kernel)) / 255.0
            blur_mask = blur_mask[...,np.newaxis] #.astype(np.float32) / 255.0
            blurred_img = cv2.blur(crop_img, (blur_kernel, blur_kernel), 0)
            crop_img = crop_img * blur_mask + blurred_img * (1 - blur_mask)
            crop_img = crop_img.astype(np.uint8)

    return crop_img, rotmat


def rotate_quad(quad, center, angle):
    rotmat = cv2.getRotationMatrix2D(center, angle, 1)
    rot_quad = cv2.transform(quad.reshape(1, -1, 2), rotmat).reshape(-1, 2)
    return rot_quad

def estimate_rotation_angle(image_data, box_np, pe, iterations=3):
    head_image = crop_head_image(image_data.copy(), box_np)
    x1, y1, w, h = box_np
    x2, y2 = x1 + w, y1 + h
    box_quad = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]]).astype(np.float32)
    box_center = np.mean(box_quad, axis=0)
    hpose = pe(head_image, isBGR=True)
    R = hpose2R(hpose)
    angle = calculate_y_angle(R)
    for i in range(iterations):
        rot_img = rotate_image(image_data.copy(), box_center, angle)[0]
        rot_head_img = crop_head_image(rot_img, box_np)
        hpose = pe(rot_head_img, isBGR=True)
        R = hpose2R(hpose)
        angle += calculate_y_angle(R)
    return angle, box_center


def get_final_crop_size(size, top_expand=0.1, left_expand=0.05, bottom_expand=0.0, right_expand=0.05):
    crop_w = int(size * (1 + left_expand + right_expand))
    crop_h = int(size * (1 + top_expand + bottom_expand))
    assert crop_w == crop_h
    return crop_w
