'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-05-25 14:26:04
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-07-20 11:43:56
FilePath: /HoloHead/bisenet/dataset.py
Description: bisenet/dataset.py
'''
import numpy as np
import os.path as osp

from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image


class BiSeNetDataset(Dataset):
    def __init__(self, img_dir, process_list, return_wh=False) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.process_list = process_list
        self.fp_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.return_wh = return_wh

    def __len__(self):
        return len(self.process_list)

    def __getitem__(self, index):
        im_name = self.process_list[index]
        image_file = osp.join(self.img_dir, im_name)
        image = Image.open(image_file)
        if self.return_wh:
            image_wh = np.array([image.width, image.height], dtype=np.int32)
        # Resize to 512x512 for BiSeNet Detection, Otherwise, it will lead bad performance
        image = image.resize((512, 512), Image.BILINEAR)
        image = np.array(image)
        image = self.fp_transform(image)
        if self.return_wh:
            return {'im_name': im_name, 'image': image, 'image_wh': image_wh}
        else:
            return {'im_name': im_name, 'image': image}


class DeepLabV3Dataset(Dataset):
    def __init__(self, img_dir, process_list, return_wh=False) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.process_list = process_list
        self.dl_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.return_wh = return_wh

    def __len__(self):
        return len(self.process_list)

    def __getitem__(self, index):
        im_name = self.process_list[index]
        image_file = osp.join(self.img_dir, im_name)
        image = Image.open(image_file)
        if self.return_wh:
            image_wh = np.array([image.width, image.height], dtype=np.int32)
        # Resize to 512x512 for BiSeNet Detection, Otherwise, it will lead bad performance
        image = image.resize((512, 512), Image.BILINEAR)
        image = np.array(image)
        image = self.dl_transform(image)
        if self.return_wh:
            return {'im_name': im_name, 'image': image, 'image_wh': image_wh}
        else:
            return {'im_name': im_name, 'image': image}


class SegmentDataset(Dataset):
    def __init__(self, img_dir, process_list, return_wh=False) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.process_list = process_list
        self.fp_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.dl_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.return_wh = return_wh

    def __len__(self):
        return len(self.process_list)

    def __getitem__(self, index):
        im_name = self.process_list[index]
        image_file = osp.join(self.img_dir, im_name)
        image = Image.open(image_file)
        if self.return_wh:
            image_wh = np.array([image.width, image.height], dtype=np.int32)
        # Resize to 512x512 for BiSeNet Detection, Otherwise, it will lead bad performance
        image = image.resize((512, 512), Image.BILINEAR)
        image = np.array(image)
        image_dl = self.dl_transform(image)
        image_fp = self.fp_transform(image)
        if self.return_wh:
            return {'im_name': im_name, 'image_dl': image_dl, 'image_fp': image_fp, 'image_wh': image_wh}
        else:
            return {'im_name': im_name, 'image_dl': image_dl, 'image_fp': image_fp}
