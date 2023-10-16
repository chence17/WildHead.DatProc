import cv2
import numpy as np


def rotate_image(img, quad, tf_quad, borderMode=cv2.BORDER_REFLECT, upsample=2):
    mat = cv2.getAffineTransform(tf_quad[:3], quad[:3])
    angle = np.degrees(np.arctan2(mat[1, 0], mat[0, 0]))
    center = np.mean(quad[:3], axis=0)
    rotmat = cv2.getRotationMatrix2D(center, angle, 1)
    crop_w = img.shape[1]
    crop_h = img.shape[0]
    crop_size = (crop_w, crop_h)
    if upsample is None or upsample == 1:
        crop_img = cv2.warpAffine(np.array(img), rotmat, crop_size, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
    else:
        assert isinstance(upsample, int)
        crop_size_large = (crop_w * upsample, crop_h * upsample)
        crop_img = cv2.warpAffine(np.array(img), upsample * rotmat, crop_size_large,
                                  flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
        crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_AREA)

    empty = np.ones_like(img) * 255
    crop_mask = cv2.warpAffine(empty, rotmat, crop_size)

    if True:
        size = min(crop_w, crop_h)
        mask_kernel = int(size*0.02)*2+1
        blur_kernel = int(size*0.03)*2+1

        if crop_mask.mean() < 255:
            blur_mask = cv2.blur(crop_mask.astype(np.float32).mean(2), (mask_kernel, mask_kernel)) / 255.0
            blur_mask = blur_mask[..., np.newaxis]  #.astype(np.float32) / 255.0
            blurred_img = cv2.blur(crop_img, (blur_kernel, blur_kernel), 0)
            crop_img = crop_img * blur_mask + blurred_img * (1 - blur_mask)
            crop_img = crop_img.astype(np.uint8)

    rot_quad = cv2.transform(quad.reshape(1, -1, 2), rotmat).reshape(-1, 2)
    return crop_img, rotmat, rot_quad


def generate_results(rotated_image, rot_quad, box_np, head_image_size):
    x1, y1 = int(rot_quad[0, 0]), int(rot_quad[0, 1])
    x2, y2 = int(rot_quad[2, 0]), int(rot_quad[2, 1])
    w, h = x2 - x1, y2 - y1
    w = h = max(w, h) # assert rot_quad is square
    head_crop_box = np.array([x1 - box_np[0], y1 - box_np[1], w, h])
    head_rot_quad = rot_quad - np.array([box_np[0], box_np[1]])
    head_image = crop_head_image(rotated_image.copy(), box_np)
    assert head_image.shape[0] == head_image.shape[1]
    scale = head_image_size / head_image.shape[0]
    head_image = cv2.resize(head_image, (head_image_size, head_image_size))
    head_crop_box = head_crop_box * scale
    head_rot_quad = head_rot_quad * scale
    return head_image, head_crop_box, head_rot_quad


def crop_head_image(image_data, box):
    # expected input: [x_min, y_min, w, h]
    # print('box', box)
    x_min, y_min, w, h = np.int32(box)
    x_max, y_max = x_min + w, y_min + h
    # print(x_min, y_min, x_max, y_max)
    top_size, bottom_size, left_size, right_size = 0, 0, 0, 0
    if x_min < 0:
        left_size = abs(x_min)
        x_min = 0
    if y_min < 0:
        top_size = abs(y_min)
        y_min = 0
    if x_max > image_data.shape[1]:
        right_size = abs(x_max - image_data.shape[1])
        x_max = image_data.shape[1]
    if y_max > image_data.shape[0]:
        bottom_size = abs(y_max - image_data.shape[0])
        y_max = image_data.shape[0]
    # print(top_size, bottom_size, left_size, right_size)
    crop_img = image_data[y_min:y_max, x_min:x_max]
    # print(crop_img.shape)
    if top_size + bottom_size + left_size + right_size != 0:
        crop_msk = np.ones_like(crop_img) * 255
        crop_img = cv2.copyMakeBorder(crop_img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
        crop_msk = cv2.copyMakeBorder(crop_msk, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT, value=0)
        size = max(crop_img.shape[:2])
        mask_kernel = int(size*0.02)*2+1
        blur_kernel = int(size*0.03)*2+1
        blur_mask = cv2.blur(crop_msk.astype(np.float32).mean(2), (mask_kernel, mask_kernel)) / 255.0
        blur_mask = blur_mask[..., np.newaxis]  # .astype(np.float32) / 255.0
        blurred_img = cv2.blur(crop_img, (blur_kernel, blur_kernel), 0)
        crop_img = crop_img * blur_mask + blurred_img * (1 - blur_mask)
        crop_img = crop_img.astype(np.uint8)
    # print(crop_img.shape, (h, w))
    assert crop_img.shape[0] == h
    assert crop_img.shape[1] == w
    return crop_img


def crop_head_parsing(image_data, box, bg_value=0):
    # expected input: [x_min, y_min, w, h]
    # print('box', box)
    x_min, y_min, w, h = np.int32(box)
    x_max, y_max = x_min + w, y_min + h
    # print(x_min, y_min, x_max, y_max)
    top_size, bottom_size, left_size, right_size = 0, 0, 0, 0
    if x_min < 0:
        left_size = abs(x_min)
        x_min = 0
    if y_min < 0:
        top_size = abs(y_min)
        y_min = 0
    if x_max > image_data.shape[1]:
        right_size = abs(x_max - image_data.shape[1])
        x_max = image_data.shape[1]
    if y_max > image_data.shape[0]:
        bottom_size = abs(y_max - image_data.shape[0])
        y_max = image_data.shape[0]
    # print(top_size, bottom_size, left_size, right_size)
    crop_img = image_data[y_min:y_max, x_min:x_max]
    # print(crop_img.shape)
    if top_size + bottom_size + left_size + right_size != 0:
        crop_img = cv2.copyMakeBorder(crop_img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT, value=bg_value)
    # print(crop_img.shape, (h, w))
    assert crop_img.shape[0] == h
    assert crop_img.shape[1] == w
    return crop_img
