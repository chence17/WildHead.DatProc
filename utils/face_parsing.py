'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-09-17 09:42:41
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-09-17 10:07:45
FilePath: /DatProc/face_parsing.py
Description: face_parsing.py
'''
import os

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
from torchvision.transforms import transforms
from skimage import measure

from bisenet.bisenet import load_BiSeNet_model
from ibug.face_parsing import FaceParser


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


def show_result(ori_img, ori_sem, label, show_axis=False):
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


def prepare(ori_img, is_bgr):
    if is_bgr:
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_h, ori_w = ori_img.shape[:2]
    if ori_h == 512 and ori_w == 512:
        img = ori_img
    elif ori_h > 512 and ori_w > 512:
        img = cv2.resize(ori_img, (512, 512), interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(ori_img, (512, 512), interpolation=cv2.INTER_LINEAR)
    return img, ori_h, ori_w, ori_img


def set_label(ori_sem, src_label, dst_label):
    src_lb2num = {lb: num for num, lb in enumerate(src_label)}
    dst_lb2num = {lb: num for num, lb in enumerate(dst_label)}
    ori_sem_ = np.zeros_like(ori_sem)
    for lb in src_label:
        ori_sem_[ori_sem == src_lb2num[lb]] = dst_lb2num[lb]
    return ori_sem_


def vote_sem(sem_list):
    sem_list_ = [i[..., None] for i in sem_list]
    all_sem = np.concatenate(sem_list_, axis=2)
    all_sem_count = np.zeros(list(all_sem.shape[:2]) + [all_sem.max() + 1], dtype=np.uint8)
    for i in range(all_sem.max() + 1):
        all_sem_count[:, :, i] = np.sum(all_sem == i, axis=2)
    sem = np.argmax(all_sem_count, axis=2).astype(np.uint8)
    return sem


def filter_small_regions(mask, threshold=0.1, kernel_size=(3, 3), close_iterations=5):
    if np.sum(mask) == 0:
        return copy.deepcopy(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    closed = cv2.morphologyEx(copy.deepcopy(mask), cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    label = measure.label(closed, connectivity=2)
    props = measure.regionprops(label)
    area_pixels = [p.area for p in props]
    area_threshold = np.mean(area_pixels) * threshold
    mask_ = np.zeros_like(mask)
    for p in props:
        if p.area >= area_threshold:
            p_mask = (label == p.label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(p_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
            p_mask_ = cv2.drawContours(np.zeros_like(p_mask), contours, 0, 255, cv2.FILLED)
            mask_[p_mask_ > 0] = 255
    return mask_


class HeadParser(object):
    def __init__(self) -> None:
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.fpp_label = ['bg', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                          'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        self.fpp_model = load_BiSeNet_model('assets/bisenet/faceparsing_model.pth', device=self.device)
        self.fpp_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.ibug_label = ['bg', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'nose', 'u_lip', 'mouth', 'l_lip',
                           'hair', 'l_ear', 'r_ear', 'eye_g']
        # self.ibug_model_01 = FaceParser(device=self.device, ckpt='assets/ibug/rtnet50-fcn-14.torch',
        #                                 encoder='rtnet50', decoder='fcn', num_classes=14)
        # self.ibug_model_02 = FaceParser(device=self.device, ckpt='assets/ibug/rtnet101-fcn-14.torch',
        #                                 encoder='rtnet101', decoder='fcn', num_classes=14)
        self.ibug_model_03 = FaceParser(device=self.device, ckpt='assets/ibug/resnet50-fcn-14.torch',
                                        encoder='resnet50', decoder='fcn', num_classes=14)
        self.ibug_model_04 = FaceParser(device=self.device, ckpt='assets/ibug/resnet50-deeplabv3plus-14.torch',
                                        encoder='resnet50', decoder='deeplabv3plus', num_classes=14)
        self.label = self.fpp_label

    def run_fpp(self, fpp_transform, fpp_model, ori_img, is_bgr, show: bool = False) -> np.ndarray:
        img, ori_h, ori_w, ori_img = prepare(ori_img, is_bgr)
        img_ts = fpp_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            sem_ts = fpp_model(img_ts)[0]
        sem = sem_ts.squeeze(0).cpu().numpy().argmax(0).astype(np.uint8)
        ori_sem = cv2.resize(sem, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        ori_sem = set_label(ori_sem, self.fpp_label, self.label)
        if show:
            show_result(ori_img, ori_sem, self.label)
        return ori_sem

    def run_ibug(self, ibug_model, ori_img, is_bgr, show: bool = False) -> np.ndarray:
        img, ori_h, ori_w, ori_img = prepare(ori_img, is_bgr)
        img_bboxes = np.array([[0, 0, 512 - 1, 512 - 1]])
        sem = ibug_model.predict_img(img, img_bboxes, rgb=True)[0].astype(np.uint8)
        ori_sem = cv2.resize(sem, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        ori_sem = set_label(ori_sem, self.ibug_label, self.label)
        if show:
            show_result(ori_img, ori_sem, self.label)
        return ori_sem

    def __call__(self, ori_img, is_bgr, show: bool = False) -> np.ndarray:
        fpp_sem = self.run_fpp(self.fpp_transform, self.fpp_model, ori_img, is_bgr, False)
        # ibug_sem_01 = self.run_ibug(self.ibug_model_01, ori_img, is_bgr, False)
        # ibug_sem_02 = self.run_ibug(self.ibug_model_02, ori_img, is_bgr, False)
        ibug_sem_03 = self.run_ibug(self.ibug_model_03, ori_img, is_bgr, False)
        ibug_sem_04 = self.run_ibug(self.ibug_model_04, ori_img, is_bgr, False)
        lb2num = {lb: num for num, lb in enumerate(self.label)}
        fpp_sem_ = fpp_sem.copy()
        fpp_sem_[fpp_sem == lb2num['hat']] = lb2num['hair']
        # ibug_sem = vote_sem([fpp_sem_, ibug_sem_01, ibug_sem_02, ibug_sem_03, ibug_sem_04])
        ibug_sem = vote_sem([fpp_sem_, ibug_sem_03, ibug_sem_04])
        for lb in ['hair']:
            fpp_sem[ibug_sem == lb2num[lb]] = lb2num[lb]
        for lb in ['ear_r', 'neck', 'neck_l', 'cloth', 'hat', 'eye_g']:
            # ibug_sem_01[fpp_sem == lb2num[lb]] = lb2num[lb]
            # ibug_sem_02[fpp_sem == lb2num[lb]] = lb2num[lb]
            ibug_sem_03[fpp_sem == lb2num[lb]] = lb2num[lb]
            ibug_sem_04[fpp_sem == lb2num[lb]] = lb2num[lb]
        # sem = vote_sem([fpp_sem, ibug_sem_01, ibug_sem_02, ibug_sem_03, ibug_sem_04])
        sem = vote_sem([fpp_sem, ibug_sem_03, ibug_sem_04]).astype(np.uint8)
        if show:
            if is_bgr:
                ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            show_result(ori_img, sem, self.label)
        return sem

if __name__ == '__main__':
    fp = HeadParser()
    # img_dir = 'assets/outputs/mh_dataset/images'
    img_dir = '../samples'
    img_names = os.listdir(img_dir)
    for img_name in img_names:
        img = cv2.imread(os.path.join(img_dir, img_name))
        res = fp(img, True, True)

    # img = cv2.imread('assets/outputs/mh_dataset/images/0_00.jpg')
    # fp.run_fpp(fp.fpp_transform, fp.fpp_model, img, True, True)
    # fp.run_ibug(fp.ibug_model_01, img, True, True)
    # fp.run_ibug(fp.ibug_model_02, img, True, True)
    # fp.run_ibug(fp.ibug_model_03, img, True, True)
    # fp.run_ibug(fp.ibug_model_04, img, True, True)
    # fp.run(img, True, True)
