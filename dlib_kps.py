'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-09-13 19:30:32
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-09-13 19:35:35
FilePath: /DatProc/dlib_kps.py
Description: dlib_kps.py
'''
import cv2
import dlib
import pickle
import numpy as np
import os

# load face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./TDDFA_V2/weights/shape_predictor_68_face_landmarks.dat")

# load images
img_dir = "./TDDFA_V2/examples/test/original"
list_dir = os.listdir(img_dir)

# new dict for keypoints
landmarks = {}

for img_name in list_dir:
    _, extension = os.path.splitext(img_name)

    # only do it for images, not .json file
    if extension == '.json':
        continue

    img_path = os.path.join(img_dir, img_name)
    image = cv2.imread(img_path)

    # gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect face
    rects = detector(gray, 1)



    for (i, rect) in enumerate(rects):
        # get keypoints
        shape = predictor(gray, rect)

        # save kps to the dict
        landmarks[img_path] = [np.array([p.x, p.y]) for p in shape.parts()]

# save the data.pkl pickle
with open('./TDDFA_V2/examples/test/data.pkl', 'wb') as f:
    pickle.dump(landmarks, f)
