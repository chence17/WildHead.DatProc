'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-10-15 16:29:15
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-10-15 16:36:52
FilePath: /DatProc/filter/size_filter.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import imagesize
import os.path as osp


class ImageSizeFilter(object):
    def __init__(self, size_thres=512) -> None:
        self.size_thres = size_thres

    def __repr__(self) -> str:
        return f'ImageSizeFilter(size_thres={self.size_thres})'

    def __call__(self, img_path: str) -> bool:
        img_w, img_h = self.get_size(img_path)
        is_small = min(img_w, img_h) < self.size_thres
        return (not is_small)

    @staticmethod
    def get_size(img_path: str):
        """Return (img_w, img_h)

        Args:
            img_path (str): image path

        Returns:
            (int, int): (img_w, img_h)
        """
        assert osp.exists(img_path), f'Image {img_path} not exists!'
        img_w, img_h = imagesize.get(img_path)
        return img_w, img_h


if __name__ == "__main__":
    img_path = '/home/chence/Research/3DHeadGen/DatProc/temp/KHairstyle2/1594.DSS268961/align_images/DSS268961-022_0.jpg'
    isf = ImageSizeFilter(size_thres=512)
    print(isf)
    print(isf.get_size(img_path))
    print(isf(img_path))
