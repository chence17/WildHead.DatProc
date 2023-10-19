'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-10-19 11:44:30
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-10-19 16:23:43
FilePath: /DatProc/RunSegSingle.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import cv2
import tqdm
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage

from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms, utils


BATCH_SIZE = 32

batchify = lambda lst, batch_size: [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]

def get_mask(model, batch, cid):
    normalized_batch = transforms.functional.normalize(
        batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    output = model(normalized_batch)['out']
    normalized_masks = torch.nn.functional.softmax(output, dim=1)
    boolean_car_masks = (normalized_masks.argmax(1) == cid)
    return boolean_car_masks.float()

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    return parser.parse_args()


def find_images(root_dir:str)-> list:
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if os.path.splitext(filename)[-1].lower() in image_extensions:
                image_paths.append(os.path.relpath(os.path.join(dirpath, filename), root_dir))
    return image_paths

args = parse_args()
# load segmentation net
seg_net = deeplabv3_resnet101(pretrained=True, progress=False).to('cuda:0')
seg_net.requires_grad_(False)
seg_net.eval()

# multiple script
image_dir = args.input_path
mask_dir = args.output_path
os.makedirs(mask_dir, exist_ok=True)

image_names = find_images(image_dir)
image_name_batches = batchify(image_names, batch_size=BATCH_SIZE)

for batch_name in tqdm.tqdm(image_name_batches):
    batch_tensors = []
    # stack images
    for image_name in batch_name:
        image_data = Image.open(os.path.join(image_dir, image_name))
        image_tensor = preprocess(image_data)
        batch_tensors.append(image_tensor)
    batch_tensors = torch.stack(batch_tensors).to('cuda:0')
    # run model
    batch_masks = get_mask(seg_net, batch_tensors, 15)
    # unstack masks
    batch_masks = torch.unbind(batch_masks, dim=0)
    for image_name, mask_tensor in zip(batch_name, batch_masks):
        # save mask
        subdir = os.path.dirname(image_name)
        os.makedirs(os.path.join(mask_dir, subdir), exist_ok=True)
        mask_image = ToPILImage()(mask_tensor)
        mask_image = cv2.resize(np.array(mask_image), (image_data.size), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(mask_dir, image_name), mask_image)

