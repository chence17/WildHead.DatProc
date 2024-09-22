import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from PIL import Image
from visualize.vis_2d import show_image
# from fastsam import FastSAM, FastSAMPrompt


class HeadSegmenter(object):
    def __init__(self, use_fsam=False) -> None:
        self.dlv3_preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        # load segmentation net
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.dlv3_net = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT, progress=False).to(self.device)
        self.dlv3_net.requires_grad_(False)
        self.dlv3_net.eval()

        self.use_fsam = use_fsam
        if self.use_fsam:
            raise NotImplementedError("FastSAM is not implemented yet.")
            # self.fsam_net = FastSAM('assets/FastSAM-x.pt')

    def __call__(self, ori_img, isBGR, show: bool = False):
        """Get head mask.

        Args:
            ori_img (np.array): Image data.
            isBGR (bool): True for BGR. False for RGB.
            show (bool, optional): _description_. Defaults to False.

        Returns:
            np.array: mask. np.uint8. 0 for background, 255 for head.
        """
        if isBGR:
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        mask = self.run_dlv3(self.dlv3_preprocess, self.dlv3_net, ori_img, isBGR=False, show=False)
        if self.use_fsam:
            box_xyxy = np.array(self.get_box_xyxy(self.get_mask_box(mask)))
            fsam_mask = self.run_fsam(self.fsam_net, ori_img, isBGR=False, box=box_xyxy, show=False)

            fsam_mask[mask == 0] = 0
            mask = fsam_mask

        if show:
            msk_image = ori_img.copy()
            msk_image[mask == 0, :] = 127
            mask_c3 = mask[:, :, None].repeat(3, axis=2)
            vis_image = np.hstack([ori_img, msk_image, mask_c3])
            show_image(vis_image, is_bgr=False, title='Final Mask')

        return mask

    @staticmethod
    def get_dlv3_mask(model, batch, cid):
        normalized_batch = transforms.functional.normalize(
            batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        output = model(normalized_batch)['out']
        # sem_classes = [
        #     '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        #     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        #     'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        # ]
        # sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
        # cid = sem_class_to_idx['car']

        normalized_masks = torch.nn.functional.softmax(output, dim=1)

        boolean_car_masks = (normalized_masks.argmax(1) == cid)
        return boolean_car_masks.float()

    @staticmethod
    def get_mask_box(mask):
        # Find contours of the mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box of the largest contour
        x1, y1, w, h = cv2.boundingRect(largest_contour)
        return [x1, y1, w, h]

    @staticmethod
    def get_box_xyxy(box):
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        return [x1, y1, x2, y2]

    def run_dlv3(self, dlv3_preprocess, dlv3_net, ori_img, isBGR, show: bool = False) -> np.ndarray:
        if isBGR:
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        ori_img_pil = Image.fromarray(ori_img)
        # Apply the transformation to the image
        img_ts = dlv3_preprocess(ori_img_pil).unsqueeze(0).to(self.device)
        # 15 means human mask
        mask0 = self.get_dlv3_mask(dlv3_net, img_ts, 15).unsqueeze(1)
        # Squeeze the tensor to remove unnecessary dimensions and convert to PIL Image
        mask_squeezed = torch.squeeze(mask0)
        mask = ToPILImage()(mask_squeezed)
        mask = cv2.resize(np.array(mask), (ori_img.shape[1], ori_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = np.array(mask)
        if show:
            msk_image = ori_img.copy()
            msk_image[mask == 0, :] = 127
            mask_c3 = mask[:, :, None].repeat(3, axis=2)
            vis_image = np.hstack([ori_img, msk_image, mask_c3])
            show_image(vis_image, is_bgr=False, title='deeplabv3_resnet101')
        return mask

    def run_fsam(self, fsam_net, ori_img, isBGR, box, show: bool = False) -> np.ndarray:
        if isBGR:
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        everything_results = fsam_net(ori_img, device=self.device, retina_masks=True, imgsz=640, conf=0.4, iou=0.9)
        prompt_process = FastSAMPrompt(ori_img, everything_results, device=self.device)
        anno = prompt_process.box_prompt(bboxes=[box])
        if len(anno) == 0 or anno is None:
            return None

        mask = np.zeros_like(anno[0], dtype=np.uint8)
        mask[anno[0] != 0] = 255

        # Find the connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Find the largest connected component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        # Create a mask for the largest connected component
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == largest_label] = 255

        if show:
            msk_image = ori_img.copy()
            msk_image[mask == 0, :] = 127
            mask_c3 = mask[:, :, None].repeat(3, axis=2)
            vis_image = np.hstack([ori_img, msk_image, mask_c3])
            show_image(vis_image, is_bgr=False, title='FastSAM')

        return mask
