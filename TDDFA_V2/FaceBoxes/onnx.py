'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-09-13 19:30:32
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-09-13 19:51:14
FilePath: /DatProc/TDDFA_V2/FaceBoxes/onnx.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# coding: utf-8

__author__ = 'cleardusk'

import torch

from TDDFA_V2.FaceBoxes.models.faceboxes import FaceBoxesNet
from TDDFA_V2.FaceBoxes.utils.functions import load_model


def convert_to_onnx(onnx_path):
    pretrained_path = onnx_path.replace('.onnx', '.pth')
    # 1. load model
    torch.set_grad_enabled(False)
    net = FaceBoxesNet(phase='test', size=None, num_classes=2)  # initialize detector
    net = load_model(net, pretrained_path=pretrained_path, load_to_cpu=True)
    net.eval()

    # 2. convert
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 720, 1080)
    # export with dynamic axes for various input sizes
    torch.onnx.export(
        net,
        (dummy_input,),
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': [0, 2, 3],
            'output': [0]
        },
        do_constant_folding=True
    )
    print(f'Convert {pretrained_path} to {onnx_path} done.')
