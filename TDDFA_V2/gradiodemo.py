'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-09-13 19:30:32
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-09-13 19:59:44
FilePath: /DatProc/TDDFA_V2/gradiodemo.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# before import, make sure FaceBoxes and Sim3DR are built successfully, e.g.,
import sys
from subprocess import call
import os
import torch

torch.hub.download_url_to_file('https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Solvay_conference_1927.jpg/1400px-Solvay_conference_1927.jpg', 'solvay.jpg')

def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except Exception as e:
        print(f"Errorrrrr: {e}!")
        
print(os.getcwd())
os.chdir("/app/FaceBoxes/utils")
print(os.getcwd())
run_cmd("python3 build.py build_ext --inplace")
os.chdir("/app/Sim3DR")
print(os.getcwd())
run_cmd("python3 setup.py build_ext --inplace")
print(os.getcwd())
os.chdir("/app/utils/asset")
print(os.getcwd())
run_cmd("gcc -shared -Wall -O3 render.c -o render.so -fPIC")
os.chdir("/app")
print(os.getcwd())


import cv2
import yaml

from TDDFA_V2.FaceBoxes import FaceBoxes
from TDDFA_V2.TDDFA import TDDFA
from TDDFA_V2.utils.render import render
from TDDFA_V2.utils.depth import depth
from TDDFA_V2.utils.pncc import pncc
from TDDFA_V2.utils.uv import uv_tex
from TDDFA_V2.utils.pose import viz_pose
from TDDFA_V2.utils.serialization import ser_to_ply, ser_to_obj
from TDDFA_V2.utils.functions import draw_landmarks, get_suffix

import matplotlib.pyplot as plt
from skimage import io
import gradio as gr

# load config
cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

# Init FaceBoxes and TDDFA, recommend using onnx flag
onnx_flag = True  # or True to use ONNX to speed up
if onnx_flag:    
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    from TDDFA_V2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    from TDDFA_V2.TDDFA_ONNX import TDDFA_ONNX

    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(**cfg)
else:
    face_boxes = FaceBoxes()
    tddfa = TDDFA(gpu_mode=False, **cfg)
    


def inference (img):
    # face detection
    boxes = face_boxes(img)
    # regress 3DMM params
    param_lst, roi_box_lst = tddfa(img, boxes)
    # reconstruct vertices and render
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
    return render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=False);    


title = "3DDFA V2"
description = "demo for 3DDFA V2. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2009.09960'>Towards Fast, Accurate and Stable 3D Dense Face Alignment</a> | <a href='https://github.com/cleardusk/3DDFA_V2'>Github Repo</a></p>"
examples = [
    ['solvay.jpg']
]
gr.Interface(
    inference, 
    [gr.inputs.Image(type="numpy", label="Input")], 
    gr.outputs.Image(type="numpy", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=examples
    ).launch()
