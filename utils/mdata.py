'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-05-16 20:04:10
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-09-14 13:08:32
FilePath: /HoloHead/utils/mdata.py
Description: utils/mdata.py
'''
import pickle as pkl
import gzip
import os
import os.path as osp
import shutil
from collections import namedtuple

MVideoFile = namedtuple("MVideoFile", ["path", "frame_num", "width", "height", "fps"])
MVideoFrames = namedtuple("MVideoFrames", ["path", "frame_num", "width", "height", "frames"])
MCamera = namedtuple("MCamera", ["model", "width", "height", "inmat", "c2w", "w2c", "bd"])
MTransMatrix = namedtuple("MTransMatrix", ["rotmat", "tvec", "tfmat"])
MCameraBound = namedtuple("MCameraBound", ["near", "far"])
MPointcloud = namedtuple("MPointcloud", ["points", "colors"])
MColmapPath = namedtuple("MColmapPath", ["images_path", "masks_path", "database_path", "sparse_path", "dense_path"])

CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "rotmat", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

MFaceBBox = namedtuple('MFaceBBox', ['top_left', 'width', 'height', 'score', 'landmarks'])
MFaceBBoxLandmarks = namedtuple('MFaceBBoxLandmarks',
                                ['right_eye', 'left_eye', 'nose_tip', 'right_mouth_corner', 'left_mouth_corner'])
MHeadBBox = namedtuple('MHeadBBox', ['top_left', 'width', 'height'])
MFaceKeyPoints = namedtuple('MFaceKeyPoints', ['points', 'scores'])


def dump_pkl(obj, path, protocol=pkl.HIGHEST_PROTOCOL) -> bool:
    with gzip.open(path, "wb") as f:
        pkl.dump(obj, f, protocol=protocol)
    return True


def load_pkl(path) -> object:
    with gzip.open(path, "rb") as f:
        return pkl.load(f)


def del_file(path):
    if osp.exists(path):
        os.remove(path)
        print(f"File {path} exists. Delete file.")
    else:
        print(f"File {path} does not exist. Skip delete.")


def del_dir(path):
    if osp.exists(path):
        shutil.rmtree(path)
        print(f"Dir {path} exists. Delete dir.")
    else:
        print(f"Dir {path} does not exist. Skip delete.")

def MCamera2dict(mc):
    mc_dict = mc._asdict()
    mc_dict['c2w'] = mc_dict['c2w']._asdict()
    mc_dict['w2c'] = mc_dict['w2c']._asdict()
    mc_dict['bd'] = mc_dict['bd']._asdict()
    return mc_dict

def convert_cams_data(cams_data):
    # convert_cams_data(cams_data)
    cams_types = ['colmap', 'nl3dmm']
    for c in cams_types:
        for k, v in cams_data[c].items():
            cams_data[c][k] = MCamera2dict(v)
    return cams_data
