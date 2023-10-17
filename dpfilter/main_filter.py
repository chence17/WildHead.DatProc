'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-10-15 16:36:23
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-10-15 16:45:01
FilePath: /DatProc/filter/main_filter.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os.path as osp
import cv2
import numpy as np
from .size_filter import ImageSizeFilter
from .blur_filter import ImageBlurFilter


class ImageFilter(object):
    def __init__(self, size_thres=512, svd_thres=0.6, lap_thres=100) -> None:
        self.size_filter = ImageSizeFilter(size_thres=size_thres)
        self.blur_filter = ImageBlurFilter(svd_thres=svd_thres, lap_thres=lap_thres)

    def __repr__(self) -> str:
        return f'ImageFilter(size_thres={self.size_filter.size_thres}, svd_thres={self.blur_filter.svd_thres}, lap_thres={self.blur_filter.lap_thres})'

    def __call__(self, img_path: str, img_data: np.array = None, isBGR: bool = None) -> bool:
        assert osp.exists(img_path), f'Image {img_path} not exists!'
        if img_data is None:
            img_data = cv2.imread(img_path)
            isBGR = True
        else:
            assert isBGR is not None, 'Please specify isBGR!'
        return (self.size_filter(img_path) and self.blur_filter(img_data, isBGR=True))

    def get_size(self, img_path: str):
        """Return (img_w, img_h)

        Args:
            img_path (str): image path

        Returns:
            (int, int): (img_w, img_h)
        """
        return self.size_filter.get_size(img_path)

    def get_blur_degree(self, img_path: str):
        """Return (svd_score, lap_score). For svd_score, the lower the better. For lap_score, the higher the better.
        svd_score: [0: clear ~ 1: blur]
        lap_score: [Low: blur ~ High: clear]

        Args:
            img_data (np.array): image data
            isBGR (bool): BGR or RGB

        Returns:
            (float, float): svd_score, lap_score.
        """
        assert osp.exists(img_path), f'Image {img_path} not exists!'
        img_data = cv2.imread(img_path)
        return self.blur_filter.get_blur_degree(img_data, isBGR=True)


if __name__ == "__main__":
    img_path = '/home/chence/Research/3DHeadGen/DatProc/temp/KHairstyle2/1594.DSS268961/align_images/DSS268961-022_0.jpg'
    img_flt = ImageFilter(size_thres=512, svd_thres=0.6, lap_thres=100)
    print(img_flt)
    print(img_flt.get_size(img_path))
    print(img_flt.get_blur_degree(img_path))
    print(img_flt(img_path))
