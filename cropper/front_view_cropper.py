
import cv2
import yaml
import numpy as np

from TDDFA_V2.FaceBoxes import FaceBoxes
from TDDFA_V2.TDDFA import TDDFA
from TDDFA_V2.utils.pose import P2sRt, matrix2angle


class FrontViewCropper(object):
    def __init__(self, config_file='TDDFA_V2/configs/mb1_120x120.yml', mode='gpu'):
        self.cfg = yaml.load(open(config_file), Loader=yaml.SafeLoader)
        self.size = 512
        # Initialize FaceBoxes and TDDFA
        self.mode = mode
        self.tddfa = TDDFA(gpu_mode=(self.mode == 'gpu'), **self.cfg)
        self.face_boxes = FaceBoxes()

    def __call__(self, image_data, landmarks):
        """Return cropped_img, camera_poses, quad, tf_quad if process sucess.

        Args:
            image_data (np.array): Image data
            landmarks (np.array): Face Landmarks

        Returns:
            (np.array, np.array, np.array, np.array): cropped_img, camera_poses, quad, tf_quad
        """
        quad, quad_c, quad_x, quad_y = self.get_crop_bound(landmarks)

        bound = np.array([[0, 0], [0, self.size-1], [self.size-1, self.size-1], [self.size-1, 0]], dtype=np.float32)
        mat = cv2.getAffineTransform(quad[:3], bound[:3])
        img = self.crop_image(image_data, mat, self.size, self.size)
        h, w = img.shape[:2]

        # Detect faces, get 3DMM params and roi boxes
        boxes = self.face_boxes(img)
        if len(boxes) == 0:
            raise ValueError(f"Length of boxes is {len(boxes)}.")

        param_lst, roi_box_lst = self.tddfa(img, boxes)
        box_idx = self.find_center_bbox(roi_box_lst, w, h)

        param = param_lst[box_idx]
        P = param[:12].reshape(3, -1)  # camera matrix
        s_relative, R, t3d = P2sRt(P)
        pose = matrix2angle(R)
        pose = [p * 180 / np.pi for p in pose]

        # Adjust z-translation in object space
        R_ = param[:12].reshape(3, -1)[:, :3]
        u = self.tddfa.bfm.u.reshape(3, -1, order='F')
        trans_z = np.array([0, 0, 0.5*u[2].mean()])  # Adjust the object center
        trans = np.matmul(R_, trans_z.reshape(3, 1))
        t3d += trans.reshape(3)

        ''' Camera extrinsic estimation for GAN training '''
        # Normalize P to fit in the original image (before 3DDFA cropping)
        sx, sy, ex, ey = roi_box_lst[0]
        scale_x = (ex - sx) / self.tddfa.size
        scale_y = (ey - sy) / self.tddfa.size
        t3d[0] = (t3d[0]-1) * scale_x + sx
        t3d[1] = (self.tddfa.size-t3d[1]) * scale_y + sy
        t3d[0] = (t3d[0] - 0.5*(w-1)) / (0.5*(w-1)) # Normalize to [-1,1]
        t3d[1] = (t3d[1] - 0.5*(h-1)) / (0.5*(h-1)) # Normalize to [-1,1], y is flipped for image space
        t3d[1] *= -1
        t3d[2] = 0  # orthogonal camera is agnostic to Z (the model always outputs 66.67)

        s_relative = s_relative * 2000
        scale_x = (ex - sx) / (w-1)
        scale_y = (ey - sy) / (h-1)
        s = (scale_x + scale_y) / 2 * s_relative
        # print(f"[{iteration}] s={s} t3d={t3d}")

        if s < 0.7 or s > 1.3:
            raise ValueError(f"s is {s}.")
        if abs(pose[0]) > 90 or abs(pose[1]) > 80 or abs(pose[2]) > 50:
            raise ValueError(f"pose is {pose}.")
        if abs(t3d[0]) > 1. or abs(t3d[1]) > 1.:
            raise ValueError(f"t3d is {t3d}.")

        quad_c = quad_c + quad_x * t3d[0]
        quad_c = quad_c - quad_y * t3d[1]
        quad_x = quad_x * s
        quad_y = quad_y * s
        c, x, y = quad_c, quad_x, quad_y
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y]).astype(np.float32)

        # final projection matrix
        s = 1
        t3d = 0 * t3d
        R[:, :3] = R[:, :3] * s
        P = np.concatenate([R, t3d[:, None]], 1)
        P = np.concatenate([P, np.array([[0, 0, 0, 1.]])], 0)
        camera_poses = self.eg3dcamparams(P.flatten())

        # Save cropped images
        cropped_img, quad, tf_quad = self.crop_final(image_data, size=self.size, quad=quad)

        return cropped_img, camera_poses, quad, tf_quad

    @staticmethod
    def get_crop_bound(lm, method="ffhq"):
        if lm.shape[0] == 106:
            left_e = lm[104]
            right_e = lm[105]
            nose = lm[49]
            left_m = lm[84]
            right_m = lm[90]
            center = (lm[1] + lm[31]) * 0.5
        elif lm.shape[0] == 68:
            left_e = np.mean(lm[36:42], axis=0)
            right_e = np.mean(lm[42:48], axis=0)
            nose = lm[33]
            left_m = lm[48]
            right_m = lm[54]
            center = (lm[0] + lm[16]) * 0.5
        else:
            raise ValueError(f"Unknown type of keypoints with a length of {len(lm)}")

        if method == "ffhq":
            eye_to_eye = right_e - left_e
            eye_avg = (left_e + right_e) * 0.5
            mouth_avg = (left_m + right_m) * 0.5
            eye_to_mouth = mouth_avg - eye_avg
            x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
            x /= np.hypot(*x)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c = eye_avg + eye_to_mouth * 0.1
        elif method == "default":
            eye_to_eye = right_e - left_e
            eye_avg = (left_e + right_e) * 0.5
            eye_to_nose = nose - eye_avg
            x = eye_to_eye.copy()
            x /= np.hypot(*x)
            x *= max(np.hypot(*eye_to_eye) * 2.4, np.hypot(*eye_to_nose) * 2.75)
            y = np.flipud(x) * [-1, 1]
            c = center
        else:
            raise ValueError('%s crop method not supported yet.' % method)
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        return quad.astype(np.float32), c, x, y

    @staticmethod
    def crop_image(img, mat, crop_w, crop_h, upsample=1, borderMode=cv2.BORDER_CONSTANT):
        crop_size = (crop_w, crop_h)
        if upsample is None or upsample == 1:
            crop_img = cv2.warpAffine(np.array(img), mat, crop_size, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
        else:
            assert isinstance(upsample, int)
            crop_size_large = (crop_w*upsample, crop_h*upsample)
            crop_img = cv2.warpAffine(np.array(img), upsample*mat, crop_size_large,
                                      flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
            crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_AREA)
        return crop_img

    @staticmethod
    def find_center_bbox(roi_box_lst, w, h):
        bboxes = np.array(roi_box_lst)
        dx = 0.5*(bboxes[:, 0] + bboxes[:, 2]) - 0.5*(w-1)
        dy = 0.5*(bboxes[:, 1] + bboxes[:, 3]) - 0.5*(h-1)
        dist = np.stack([dx, dy], 1)
        return np.argmin(np.linalg.norm(dist, axis=1))

    @staticmethod
    def eg3dcamparams(R_in):
        # World Coordinate System: x(right), y(up), z(forward)
        camera_dist = 2.7
        intrinsics = np.array([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]])
        # assume inputs are rotation matrices for world2cam projection
        R = np.array(R_in).astype(np.float32).reshape(4,4)
        # add camera translation
        t = np.eye(4, dtype=np.float32)
        t[2, 3] = - camera_dist

        # convert to OpenCV camera
        convert = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]).astype(np.float32)

        # world2cam -> cam2world
        P = convert @ t @ R
        cam2world = np.linalg.inv(P)

        # add intrinsics
        label_new = np.concatenate([cam2world.reshape(16), intrinsics.reshape(9)], -1)
        return label_new

    @staticmethod
    def crop_final(
        img,
        size=512,
        quad=None,
        top_expand=0.1,
        left_expand=0.05,
        bottom_expand=0.0,
        right_expand=0.05,
        blur_kernel=None,
        borderMode=cv2.BORDER_REFLECT,
        upsample=2,
        min_size=256,
    ):
        orig_size = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
        if min_size is not None and orig_size < min_size:
            raise ValueError(f"min_size is {min_size} and orig_size is {orig_size}.")

        crop_w = int(size * (1 + left_expand + right_expand))
        crop_h = int(size * (1 + top_expand + bottom_expand))
        crop_size = (crop_w, crop_h)

        top = int(size * top_expand)
        left = int(size * left_expand)
        size -= 1
        bound = np.array([[left, top], [left, top + size], [left + size, top + size], [left + size, top]],
                         dtype=np.float32)
        tf_quad = np.array([[0, 0], [0, crop_h-1], [crop_w-1, crop_h-1], [crop_w-1, 0]], dtype=np.float32)

        mat = cv2.getAffineTransform(quad[:3], bound[:3])

        # Calculate the inverse of the affine transformation matrix
        invmat = cv2.invertAffineTransform(mat)
        # Apply the inverse transformation matrix to the transformed points
        orig_quad = cv2.transform(tf_quad.reshape(1, -1, 2), invmat).reshape(-1, 2)

        if upsample is None or upsample == 1:
            crop_img = cv2.warpAffine(np.array(img), mat, crop_size, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
        else:
            assert isinstance(upsample, int)
            crop_size_large = (crop_w*upsample, crop_h*upsample)
            crop_img = cv2.warpAffine(np.array(img), upsample*mat, crop_size_large,
                                      flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
            crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_AREA)

        empty = np.ones_like(img) * 255
        crop_mask = cv2.warpAffine(empty, mat, crop_size)

        mask_kernel = int(size*0.02)*2+1
        blur_kernel = int(size*0.03)*2+1 if blur_kernel is None else blur_kernel
        # downsample_size = (crop_w//8, crop_h//8)

        if crop_mask.mean() < 255:
            blur_mask = cv2.blur(crop_mask.astype(np.float32).mean(2), (mask_kernel, mask_kernel)) / 255.0
            blur_mask = blur_mask[..., np.newaxis]  #.astype(np.float32) / 255.0
            blurred_img = cv2.blur(crop_img, (blur_kernel, blur_kernel), 0)
            crop_img = crop_img * blur_mask + blurred_img * (1 - blur_mask)
            crop_img = crop_img.astype(np.uint8)

        return crop_img, orig_quad, tf_quad
