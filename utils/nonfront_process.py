import gzip
import pickle as pkl
import tqdm
import math
import numpy as np
import cv2
import os
import os.path as osp
import numpy as np
import onnxruntime
import numba as nb
from scipy.spatial.transform import Rotation

HOLO_DIR = '/home/chence/Research/3DHeadGen/HoloHead'

def load_pkl(path) -> object:
    with gzip.open(path, "rb") as f:
        return pkl.load(f)

def dump_pkl(obj, path, protocol=pkl.HIGHEST_PROTOCOL) -> bool:
    with gzip.open(path, "wb") as f:
        pkl.dump(obj, f, protocol=protocol)
    return True

def filter_ibbox_data(ibbox_data, thres=512. * 512.):
    filter_dict = {}
    for i in tqdm.tqdm(ibbox_data.keys()):
        if ibbox_data[i] is None:
            continue
        if (ibbox_data[i][2] * ibbox_data[i][3]) > thres:
            filter_dict[i] = ibbox_data[i]
    return filter_dict

def filter_fkpts_data(fkpts_data):
    filter_dict = {}
    for i in tqdm.tqdm(fkpts_data.keys()):
        if fkpts_data[i] is None:
            continue
        filter_dict[i] = fkpts_data[i]
    return filter_dict

def filter_hpose_data(hpose_data):
    filter_dict = {}
    for i in tqdm.tqdm(hpose_data.keys()):
        if hpose_data[i] is None:
            continue
        filter_dict[i] = hpose_data[i]
    return filter_dict

def rotate_point(point, center, angle):
    """
    计算点绕固定点旋转一定角度后的新坐标
    :param point: 要旋转的点的坐标 (x, y)
    :param center: 固定点的坐标 (x, y)
    :param angle: 旋转角度（弧度制）
    :return: 旋转后的新坐标 (x', y')
    """
    x, y = point
    cx, cy = center

    # 将角度转换为弧度
    angle_rad = math.radians(angle)

    # 计算旋转后的新坐标
    new_x = (x - cx) * math.cos(angle_rad) - (y - cy) * math.sin(angle_rad) + cx
    new_y = (x - cx) * math.sin(angle_rad) + (y - cy) * math.cos(angle_rad) + cy

    return new_x, new_y

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
        return None

    crop_w = int(size * (1 + left_expand + right_expand))
    crop_h = int(size * (1 + top_expand + bottom_expand))
    crop_size = (crop_w, crop_h)

    top = int(size * top_expand)
    left = int(size * left_expand)
    size -= 1
    bound = np.array([[left, top], [left, top + size], [left + size, top + size], [left + size, top]],
                        dtype=np.float32)

    mat = cv2.getAffineTransform(quad[:3], bound[:3])
    if upsample is None or upsample == 1:
        crop_img = cv2.warpAffine(np.array(img), mat, crop_size, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
    else:
        assert isinstance(upsample, int)
        crop_size_large = (crop_w*upsample,crop_h*upsample)
        crop_img = cv2.warpAffine(np.array(img), upsample*mat, crop_size_large, flags=cv2.INTER_LANCZOS4, borderMode=borderMode)
        crop_img = cv2.resize(crop_img, crop_size, interpolation=cv2.INTER_AREA) 

    empty = np.ones_like(img) * 255
    crop_mask = cv2.warpAffine(empty, mat, crop_size)



    if True:
        mask_kernel = int(size*0.02)*2+1
        blur_kernel = int(size*0.03)*2+1 if blur_kernel is None else blur_kernel
        downsample_size = (crop_w//8, crop_h//8)

        if crop_mask.mean() < 255:
            blur_mask = cv2.blur(crop_mask.astype(np.float32).mean(2),(mask_kernel,mask_kernel)) / 255.0
            blur_mask = blur_mask[...,np.newaxis]#.astype(np.float32) / 255.0
            blurred_img = cv2.blur(crop_img, (blur_kernel, blur_kernel), 0)
            crop_img = crop_img * blur_mask + blurred_img * (1 - blur_mask)
            crop_img = crop_img.astype(np.uint8)

    return crop_img

@nb.njit('i8[:](f4[:,:],f4[:], f4, b1)', fastmath=True, cache=True)
def nms_cpu(boxes, confs, nms_thresh, min_mode):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]
    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]
        keep.append(idx_self)
        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)
        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    return np.array(keep)

def detect_head(img_data, yolov4_head, yolov4_head_H, yolov4_head_W, yolov4_head_input_name, yolov4_head_output_names, conf_thresh=0.5, nms_thresh=0.5):
    yolov4_head_size = (yolov4_head_H, yolov4_head_W)
    img_size = img_data.shape[:2]
    img_scale = min(yolov4_head_size[1] / img_size[1], yolov4_head_size[0] / img_size[0])
    new_h = int(img_size[0] * img_scale)
    new_w = int(img_size[1] * img_scale)
    assert new_h > 0 and new_w > 0
    assert new_h <= img_size[0] and new_w <= img_size[1]
    img_data_scaled = cv2.resize(img_data, (new_w, new_h))
    pad_top = (yolov4_head_size[0] - new_h) // 2
    pad_bottom = yolov4_head_size[0] - new_h - pad_top
    pad_left = (yolov4_head_size[1] - new_w) // 2
    pad_right = yolov4_head_size[1] - new_w - pad_left
    img_data_padded = cv2.copyMakeBorder(img_data_scaled,
                                        pad_top,
                                        pad_bottom,
                                        pad_left,
                                        pad_right,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=0)
    assert img_data_padded.shape[:2] == yolov4_head_size
    img_data_rgb = img_data_padded[..., ::-1]
    img_data_chw = img_data_rgb.transpose(2, 0, 1)
    img_data_chw = np.asarray(img_data_chw / 255., dtype=np.float32)
    img_data_nchw = img_data_chw[np.newaxis, ...]
    img_data_boxes, img_data_confs = yolov4_head.run(output_names=yolov4_head_output_names,
                                                    input_feed={yolov4_head_input_name: img_data_nchw})
    img_data_boxes = img_data_boxes[0][:, 0, :]
    img_data_confs = img_data_confs[0][:, 0]
    argwhere = img_data_confs > conf_thresh
    img_data_boxes = img_data_boxes[argwhere, :]
    img_data_confs = img_data_confs[argwhere]
    img_data_heads = []
    img_data_keep = nms_cpu(boxes=img_data_boxes, confs=img_data_confs, nms_thresh=nms_thresh, min_mode=False)
    assert img_data_keep.size > 0, 'No object detected!'
    width = img_data_padded.shape[1]
    height = img_data_padded.shape[0]
    if (img_data_keep.size > 0):
        img_data_boxes = img_data_boxes[img_data_keep, :]
        img_data_confs = img_data_confs[img_data_keep]
        for k in range(img_data_boxes.shape[0]):
            img_data_heads.append([
                img_data_boxes[k, 0] * width,
                img_data_boxes[k, 1] * height,
                img_data_boxes[k, 2] * width,
                img_data_boxes[k, 3] * height,
                img_data_confs[k],
            ])
    assert len(img_data_heads) == 1, 'Only one head is supported!'
    img_data_head = img_data_heads[0]
    x_min = (img_data_head[0] - pad_left) / img_scale
    y_min = (img_data_head[1] - pad_top) / img_scale
    x_max = (img_data_head[2] - pad_left) / img_scale
    y_max = (img_data_head[3] - pad_top) / img_scale
    # enlarge the bbox to include more background margin
    y_min = max(0, y_min - abs(y_min - y_max) / 10)
    y_max = min(img_data.shape[0], y_max + abs(y_min - y_max) / 10)
    y_center = (y_min + y_max) / 2
    y_delta = (y_max - y_min) / 2
    x_min = max(0, x_min - abs(x_min - x_max) / 5)
    x_max = min(img_data.shape[1], x_max + abs(x_min - x_max) / 5)
    x_max = min(x_max, img_data.shape[1])
    x_center = (x_min + x_max) / 2
    x_delta = (x_max - x_min) / 2
    xy_delta = max(x_delta, y_delta)
    y_min = max(0, y_center - xy_delta)
    y_max = min(img_data.shape[0], y_center + xy_delta)
    x_min = max(0, x_center - xy_delta)
    x_max = min(img_data.shape[1], x_center + xy_delta)
    img_head_width = (x_max - x_min)
    img_head_height = (y_max - y_min)
    img_head_width = max(img_head_width, img_head_height)
    img_head_height = img_head_width
    img_head_cx = ((x_min + x_max) / 2.)
    img_head_cy = ((y_min + y_max) / 2.)
    return [img_head_cx, img_head_cy, img_head_width, img_head_height]

def detect_pose(img_ybbox_crp, whenet, whenet_output_names, whenet_input_name):
    # bgr --> rgb
    rgb = img_ybbox_crp[..., ::-1]
    # hwc --> chw
    chw = rgb.transpose(2, 0, 1)
    # chw --> nchw
    nchw = np.asarray(chw[np.newaxis, :, :, :], dtype=np.float32)

    yaw = 0.0
    pitch = 0.0
    roll = 0.0
    outputs = whenet.run(
        output_names = whenet_output_names,
        input_feed = {whenet_input_name: nchw}
    )
    yaw = outputs[0][0][0]
    roll = outputs[0][0][1]
    pitch = outputs[0][0][2]
    yaw, pitch, roll = np.squeeze([yaw, pitch, roll])
    return [yaw, pitch, roll]

def eg3dcamparams(R_in):
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


fkpts_file = '/home/chence/Research/3DHeadGen/HoloHead/data/Web/DataSelected_fkpts.pkl'
hpose_file = '/home/chence/Research/3DHeadGen/HoloHead/data/Web/DataSelected_hpose.pkl'
ibbox_file = '/home/chence/Research/3DHeadGen/HoloHead/data/Web/DataSelected_ibbox.pkl'
fmeta_file = '/home/chence/Research/3DHeadGen/HoloHead/data/Web/FrontalData_meta.pkl'
fquad_file = '/home/chence/Research/3DHeadGen/HoloHead/data/Web/FrontalData_quad.pkl'
fpose_file = '/home/chence/Research/3DHeadGen/HoloHead/data/Web/FrontalData_pose.pkl'

fkpts_data = load_pkl(fkpts_file)
hpose_data = load_pkl(hpose_file)
ibbox_data = load_pkl(ibbox_file)
fmeta_data = load_pkl(fmeta_file)
fquad_data = load_pkl(fquad_file)
fpose_data = load_pkl(fpose_file)

fkpts_data_ = filter_fkpts_data(fkpts_data)
hpose_data_ = filter_hpose_data(hpose_data)
ibbox_data_ = filter_ibbox_data(ibbox_data)

fkpts_valid_keys = sorted(set(ibbox_data_.keys()).intersection(set(fkpts_data_.keys())))
print(len(fkpts_valid_keys))
hpose_valid_keys = sorted(set(ibbox_data_.keys()).intersection(set(hpose_data_.keys())))
print(len(hpose_valid_keys))
backi_valid_keys = sorted(set(hpose_valid_keys) - set(fkpts_valid_keys))
print(len(backi_valid_keys))
froni_valid_keys = sorted(set(fkpts_valid_keys) - set(fquad_data.keys()))
print(len(froni_valid_keys))
yolo_trans = load_pkl('/home/chence/Research/3DHeadGen/HoloHead/data/Web/YoloDetectTrans.pkl')
print(yolo_trans)

img_dir = '/home/chence/Research/3DHeadGen/HoloHead/data/Web/DataSelected'
size = 512

idx_tensor_yaw = [np.array(idx, dtype=np.float32) for idx in range(120)]
idx_tensor = [np.array(idx, dtype=np.float32) for idx in range(66)]

yolov4_head_H = 480
yolov4_head_W = 640
whenet_H = 224
whenet_W = 224

# YOLOv4-Head
yolov4_model_name = 'yolov4_headdetection'
yolov4_head = onnxruntime.InferenceSession(
    osp.join(HOLO_DIR, 'assets/whenet', f'saved_model_{whenet_H}x{whenet_W}/{yolov4_model_name}_{yolov4_head_H}x{yolov4_head_W}.onnx'),
    providers=[
        'CUDAExecutionProvider',
        'CPUExecutionProvider',
    ]
)
yolov4_head_input_name = yolov4_head.get_inputs()[0].name
yolov4_head_output_names = [output.name for output in yolov4_head.get_outputs()]
yolov4_head_output_shapes = [output.shape for output in yolov4_head.get_outputs()]
assert yolov4_head_output_shapes[0] == [1, 18900, 1, 4] # boxes[N, num, classes, boxes]
assert yolov4_head_output_shapes[1] == [1, 18900, 1]    # confs[N, num, classes]

# WHENet
whenet_input_name = None
whenet_output_names = None
whenet_output_shapes = None
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
whenet = onnxruntime.InferenceSession(
    osp.join(HOLO_DIR, 'assets/whenet', f'saved_model_{whenet_H}x{whenet_W}/whenet_1x3x224x224_prepost.onnx'),
    providers=[
        'CUDAExecutionProvider',
        'CPUExecutionProvider',
    ]
)
whenet_input_name = whenet.get_inputs()[0].name
whenet_output_names = [output.name for output in whenet.get_outputs()]

exec_net = None
input_name = None
conf_thresh = 0.60
nms_thresh = 0.50
save_dir = '/home/chence/Research/3DHeadGen/HoloHead/data/Web/NonFrontalData'
os.makedirs(save_dir, exist_ok=True)
results_quad = {}
results_meta = {}
results_pose = {}

for img_name in tqdm.tqdm(backi_valid_keys):
    img_path = osp.join(img_dir, img_name)
    try:
        img_data = cv2.imread(img_path)
        img_ybbox = i(img_data.copy(), yolov4_head, yolov4_head_H, yolov4_head_W, yolov4_head_input_name, yolov4_head_output_names, conf_thresh, nms_thresh)
        img_yc = np.array([img_ybbox[0], img_ybbox[1]])
        img_yw, img_yh = img_ybbox[2], img_ybbox[3]
        img_ybbox_quad = np.array([
            [img_yc[0] - img_yw / 2., img_yc[1] - img_yh / 2.],
            [img_yc[0] - img_yw / 2., img_yc[1] + img_yh / 2.],
            [img_yc[0] + img_yw / 2., img_yc[1] + img_yh / 2.],
            [img_yc[0] + img_yw / 2., img_yc[1] - img_yh / 2.]
        ])

        assert whenet_H == whenet_W
        img_ybbox_crp = crop_final(img_data.copy(), size=whenet_H, quad=img_ybbox_quad.astype(np.float32), borderMode=cv2.BORDER_REPLICATE)
        img_ybbox_crp = cv2.resize(img_ybbox_crp, (whenet_W, whenet_H), interpolation=cv2.INTER_LINEAR)
        img_hpose = detect_pose(img_ybbox_crp, whenet, whenet_output_names, whenet_input_name)

        img_qw = img_yw * yolo_trans['width_scale']
        img_qh = img_yh * yolo_trans['height_scale']
        img_qw = max(img_qw, img_qh) # 这里需要结合之后的Crop看一下是取Min还是Max
        img_qh = img_qw
        # img_qc = img_yc
        img_qcx = yolo_trans['center_x_delta_scaled'] * img_yw + img_yc[0]
        img_qcy = yolo_trans['center_y_delta_scaled'] * img_yh + img_yc[1]
        img_qc = np.array([img_qcx, img_qcy])
        img_qbbox = np.array([
            [img_qc[0] - img_qw / 2., img_qc[1] - img_qh / 2.],
            [img_qc[0] - img_qw / 2., img_qc[1] + img_qh / 2.],
            [img_qc[0] + img_qw / 2., img_qc[1] + img_qh / 2.],
            [img_qc[0] + img_qw / 2., img_qc[1] - img_qh / 2.]
        ])
        img_qbbox_enlarged = np.array([
            [img_qc[0] - img_yw / 2., img_qc[1] - img_yh / 2.],
            [img_qc[0] - img_yw / 2., img_qc[1] + img_yh / 2.],
            [img_qc[0] + img_yw / 2., img_qc[1] + img_yh / 2.],
            [img_qc[0] + img_yw / 2., img_qc[1] - img_yh / 2.]
        ])
        # img_hpose = hpose_data_[img_name]
        rot_angle = img_hpose[2]
        # rot_angle = 0
        # print(rot_angle)
        img_qbbox_ = []
        for i in img_qbbox:
            i_rot = rotate_point(i, img_yc, rot_angle)
            img_qbbox_.append(i_rot)
        img_qbbox_ = np.array(img_qbbox_)

        img_qbbox_enlarged_ = []
        for i in img_qbbox_enlarged:
            i_rot = rotate_point(i, img_yc, rot_angle)
            img_qbbox_enlarged_.append(i_rot)
        img_qbbox_enlarged_ = np.array(img_qbbox_enlarged_)

        cropped_img = crop_final(img_data, size=size, quad=img_qbbox_.astype(np.float32))
        cv2.imwrite(osp.join(save_dir, img_name), cropped_img)

        assert whenet_H == whenet_W
        cropped_img_enlarged = crop_final(img_data.copy(), size=whenet_H, quad=img_qbbox_enlarged_.astype(np.float32), borderMode=cv2.BORDER_REPLICATE)
        cropped_img_enlarged = cv2.resize(cropped_img_enlarged, (whenet_W, whenet_H), interpolation=cv2.INTER_LINEAR)
        img_hpose_enlarged = detect_pose(cropped_img_enlarged, whenet, whenet_output_names, whenet_input_name)

        yaw, pitch, roll = img_hpose_enlarged
        r_pitch, r_yaw, r_roll = -pitch, -yaw, -roll
        s = 1
        t3d = np.array([0., 0., 0.])
        R = Rotation.from_euler('zyx', [r_roll, r_yaw, r_pitch], degrees=True).as_matrix()
        R[:,:3] = R[:,:3] * s
        P = np.concatenate([R,t3d[:,None]],1)
        P = np.concatenate([P, np.array([[0,0,0,1.]])],0)
        results_pose[img_name] = P
        results_meta[img_name] = eg3dcamparams(P.flatten())
        results_quad[img_name] = img_qbbox_
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

dump_pkl(results_quad, save_dir+'_quad.pkl') # [tl, bl, br, tr]
dump_pkl(results_meta, save_dir+'_meta.pkl') # [Ext(4x4, 16) Int(3x3, 9)]
dump_pkl(results_pose, save_dir+'_pose.pkl')
