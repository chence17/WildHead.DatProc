import cv2
import numpy as np

def resize_propper(image, max_size = 512):
    scale = 512 / np.max(image.shape[:2])
    image = cv2.resize(image, (0,0), fx=scale, fy=scale)