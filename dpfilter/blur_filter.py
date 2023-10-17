import cv2
import numpy as np


class ImageBlurFilter(object):
    def __init__(self, svd_thres=0.6, lap_thres=100) -> None:
        self.lap_thres = lap_thres
        self.svd_thres = svd_thres

    def __call__(self, img_data: np.array, isBGR: bool) -> bool:
        """Return True if clear, False if blur

        Args:
            img_data (np.array): image data
            isBGR (bool): BGR or RGB

        Returns:
            bool: True if clear, False if blur
        """
        svd_score, lap_score = self.get_blur_degree(img_data, isBGR)
        return ((svd_score < self.svd_thres) and (lap_score > self.lap_thres))

    def __repr__(self) -> str:
        return f'ImageBlurFilter(svd_thres={self.svd_thres}, lap_thres={self.lap_thres})'

    def get_blur_degree(self, img_data: np.array, isBGR: bool):
        """Return (svd_score, lap_score). For svd_score, the lower the better. For lap_score, the higher the better.
        svd_score: [0: clear ~ 1: blur]
        lap_score: [Low: blur ~ High: clear]

        Args:
            img_data (np.array): image data
            isBGR (bool): BGR or RGB

        Returns:
            (float, float): svd_score, lap_score.
        """
        if isBGR:
            img_data_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        else:
            img_data_gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        svd_score = self.get_blur_degree_svd(img_data_gray)
        lap_score = self.get_blur_degree_laplacian(img_data_gray)
        return svd_score, lap_score

    @staticmethod
    def get_blur_degree_svd(img_data_gray: np.array, sv_num: int = 10):
        """
        Modified from https://github.com/fled/blur_detection/blur_detection.py
        [0: clear ~ 1: blur]
        """
        u, s, v = np.linalg.svd(img_data_gray)
        top_sv = np.sum(s[0:sv_num])
        total_sv = np.sum(s)
        score = top_sv/total_sv
        return score

    @staticmethod
    def get_blur_degree_laplacian(img_data_gray: np.array):
        """
        Modified from https://github.com/WillBrennan/BlurDetection2/process.py
        Low: blur
        High: clear
        """
        blur_map = cv2.Laplacian(img_data_gray, cv2.CV_64F)
        score = np.var(blur_map)
        return score


if __name__ == "__main__":
    import os
    img_path = '/home/chence/Research/3DHeadGen/DatProc/temp/KHairstyle2/1594.DSS268961/align_images/DSS268961-022_0.jpg'
    assert os.path.exists(img_path), f'Image {img_path} not exists!'
    img_data = cv2.imread(img_path)
    ibf = ImageBlurFilter(svd_thres=0.6, lap_thres=100)
    print(ibf)
    print(ibf.get_blur_degree(img_data, isBGR=True))
    print(ibf(img_data, isBGR=True))
