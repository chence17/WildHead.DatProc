'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-10-15 17:17:37
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-10-17 14:43:53
FilePath: /DatProc/visualize/vis_2d.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def draw_detection_box(image_data, box, text):
    # Determine the detection box coordinates
    x, y, width, height = box
    scale = math.ceil(math.sqrt(image_data.shape[0] * image_data.shape[1]) / 512)
    thinkness = scale * 2
    font_scale = scale * 0.9
    offset_scale = scale * 10

    # Draw the detection box
    cv2.rectangle(image_data, (x, y), (x + width, y + height), (0, 255, 0), thinkness)

    # Add a text label
    cv2.putText(image_data, text, (x, y - offset_scale), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 255, 0), thinkness)
    return image_data


def draw_facial_landmarks(image_data, landmarks, color=(0, 255, 0)):
    scale = math.ceil(math.sqrt(image_data.shape[0] * image_data.shape[1]) / 512)
    radius = scale * 2
    for landmark in landmarks:
        x, y = landmark
        cv2.circle(image_data, (x, y), radius, color, -1)
    return image_data


def draw_quad(image_data, quad, color=(0, 255, 0)):
    scale = math.ceil(math.sqrt(image_data.shape[0] * image_data.shape[1]) / 512)
    thickness = scale * 2
    cv2.polylines(image_data, [quad], isClosed=True, color=color, thickness=thickness)
    return image_data


def show_image(img, is_bgr, title, show_axis=False):
    if is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    if not show_axis:
        plt.axis('off')
    else:
        plt.axis('on')
    plt.show()


def show_parsing_result(ori_img, ori_sem, label, show_axis=False):
    ul = sorted(np.unique(ori_sem))
    print(ul)
    print(ori_sem.dtype, len(ul))
    n_cols = 3
    n_rows = int(np.ceil((len(label)+2) / n_cols))
    lb2num = {lb: num for num, lb in enumerate(label)}
    axis_str = 'on' if show_axis else 'off'
    plt.figure(figsize=(2*n_cols, 2*n_rows))
    plt.subplot(n_rows, n_cols, 1), plt.title('origin'), plt.imshow(ori_img), plt.axis(axis_str)
    all_mask = np.zeros_like(ori_img)
    for idx, lb in enumerate(label):
        cur_mask = np.zeros_like(ori_img)
        cur_mask[ori_sem == lb2num[lb], :] = 255
        if lb2num[lb] != 0:
            all_mask[ori_sem == lb2num[lb], :] = 255
        plt.subplot(n_rows, n_cols, idx+2), plt.title(lb), plt.imshow(cur_mask), plt.axis(axis_str)
    plt.subplot(n_rows, n_cols, len(label)+2), plt.title('all_mask'), plt.imshow(all_mask), plt.axis(axis_str)
    plt.show()


def show_parsing_result_colorized(ori_img, ori_sem, label, show_axis=False):
    def bgr_color(v):
        return (np.array(plt.cm.tab20(v)[:3]) * 255).astype(np.uint8)[::-1]

    ul = sorted(np.unique(ori_sem))
    print(ul)
    print(ori_sem.dtype, len(ul))

    colorwithlabel = []
    colorized = np.zeros((ori_sem.shape[0], ori_sem.shape[1], 3), dtype=np.uint8)
    for c in np.unique(ori_sem):
        cur_color = bgr_color(c)
        colorized[ori_sem == c] = cur_color
        colorvis = np.ones((100, 100, 3), dtype=np.uint8) * cur_color
        colorvis = cv2.putText(colorvis, f"{c}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        if label is not None:
            colorvis = cv2.putText(colorvis, f"{label[c]}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        colorwithlabel.append(colorvis)
    colorwithlabel = np.concatenate(colorwithlabel, axis=1)
    colorizedimage = (ori_img * 0.5 + colorized * 0.5).astype(np.uint8)
    concatedimages = np.concatenate([ori_img, colorizedimage, colorized], axis=1)
    cwl_h, cwl_w = colorwithlabel.shape[:2]
    cim_h, cim_w = concatedimages.shape[:2]
    colorwithlabel = cv2.resize(colorwithlabel, (cim_w, cwl_h * cim_w // cwl_w))
    final_image = np.concatenate([concatedimages, colorwithlabel], axis=0)
    # show image
    plt.figure(figsize=(20, 20))
    plt.imshow(final_image)
    plt.axis('off' if not show_axis else 'on')
    plt.show()
