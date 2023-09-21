import os
import re
import cv2
import numpy as np


class ProcessError(Exception):
    def __init__(self, message="Process Error"):
        self.message = message
        super().__init__(self.message)

def calc_h2b_ratio(mask_data: np.ndarray) -> float:
    h, w = mask_data.shape
    assert h == w, "image not suqare"
    output = cv2.connectedComponentsWithStats(mask_data, connectivity=4)[1]
    unique, counts = np.unique(output, return_counts=True)
    unique, counts = unique[1:], counts[1:] # discard background
    max_idx = np.argmax(counts)
    result = np.where(output == unique[max_idx], 255, 0).astype(np.uint8)
    head_top = np.argmax(np.any(result, axis=1))
    half_h = h/2
    ratio = max(half_h/(half_h-head_top), 1)
    return ratio

def find_meta_files(root_dir):
    pattern = r'meta_(\d+)-(\d+)\.json'
    json_file_paths = []
    # Loop through the files in the folder
    for filename in os.listdir(root_dir):
        match = re.match(pattern, filename)
        if match: json_file_paths.append(os.path.join(root_dir, filename))
    return json_file_paths

def get_cam_sperical_coord(img_meta):
    c2w = np.array(img_meta["head"]["00"]["camera"][:16]).reshape(4, 4)
    T = c2w[:3, 3]
    x, y, z = T
    theta = np.rad2deg(np.arctan(np.sqrt(x**2+z**2)/y))
    phi = np.rad2deg(np.arctan(z/x))+180
    return theta, phi, x, y, z

if __name__ == '__main__':
    test_mask = cv2.imread("/home/shitianhao/project/DatProc/assets/mask.jpg", cv2.IMREAD_GRAYSCALE)
    _, nm = calc_h2b_ratio(test_mask)
    cv2.imwrite("/home/shitianhao/project/DatProc/assets/nm.jpg", nm)
        