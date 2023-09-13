'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-09-13 19:30:32
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-09-13 19:47:39
FilePath: /DatProc/TDDFA_V2/bfm/bfm.py
Description: bfm.py
'''
# coding: utf-8

from TDDFA_V2.utils.io import _load
import numpy as np
import os.path as osp
__author__ = 'cleardusk'


def make_abs_path(fn): return osp.join(osp.dirname(osp.realpath(__file__)), fn)


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


class BFMModel(object):
    def __init__(self, bfm_fp, shape_dim=40, exp_dim=10):
        bfm = _load(bfm_fp)
        self.u = bfm.get('u').astype(np.float32)  # fix bug
        self.w_shp = bfm.get('w_shp').astype(np.float32)[..., :shape_dim]
        self.w_exp = bfm.get('w_exp').astype(np.float32)[..., :exp_dim]
        if osp.split(bfm_fp)[-1] == 'bfm_noneck_v3.pkl':
            # this tri/face is re-built for bfm_noneck_v3
            self.tri = _load(make_abs_path('../configs/tri.pkl'))
        else:
            self.tri = bfm.get('tri')

        self.tri = _to_ctype(self.tri.T).astype(np.int32)
        self.keypoints = bfm.get('keypoints').astype(np.int64)  # fix bug
        w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(w, axis=0)

        self.u_base = self.u[self.keypoints].reshape(-1, 1)
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]
