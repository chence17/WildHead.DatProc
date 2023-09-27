'''
Author: tianhao 120090472@link.cuhk.edu.cn
Date: 2023-09-26 09:58:29
LastEditors: tianhao 120090472@link.cuhk.edu.cn
LastEditTime: 2023-09-26 16:28:53
FilePath: /DatProc/utils/cam_pose_utils.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import numpy as np

def get_cam_coords(c2w):
    # World Coordinate System: x(right), y(up), z(forward)
    T = c2w[:3, 3]
    x, y, z = T
    r = np.sqrt(x**2+y**2+z**2)
    # theta = np.rad2deg(np.arctan(np.sqrt(x**2+z**2)/y))
    theta = np.rad2deg(np.arctan2(x, z))
    if theta >= -90 and theta <= 90:
        theta += 90
    elif theta>=-180 and theta < -90:
        theta += 90
    elif theta>90 and theta <= 180:
        theta -= 270
    else:
        raise ValueError('theta out of range')
    # phi = np.rad2deg(np.arctan(z/x))+180
    phi = np.rad2deg(np.arccos(y/r))
    return [theta, phi, r, x, y, z] # [:3] sperical cood, [3:] cartesian cood

def spherical_to_cartesian(theta, phi, r=1):
    x = -r*np.sin(np.deg2rad(phi))*np.cos(np.deg2rad(theta))
    z = r*np.sin(np.deg2rad(phi))*np.sin(np.deg2rad(theta))
    y = r*np.cos(np.deg2rad(phi))
    return x, y, z

def cam_coord_to_cam_matrix(x, y, z):
    T = np.array([z, x, y])
    down_vec = np.array([0, 0, -1])
    cam_z = -T/np.linalg.norm(T)
    cam_x = np.cross(down_vec, cam_z)
    cam_x = cam_x/np.linalg.norm(cam_x)
    cam_y = np.cross(cam_z, cam_x)
    cam_y = cam_y/np.linalg.norm(cam_y)
    R = np.dstack([cam_x, cam_y, cam_z]).squeeze()
    return np.hstack([R, T.reshape(-1, 1)])

def gen_cam_matrix(theta, phi, r=1):
    x, y, z = spherical_to_cartesian(theta, phi, r)
    RT = cam_coord_to_cam_matrix(x, y, z)
    bottom = np.array([0, 0, 0, 1])
    c2w = np.vstack([RT, bottom])
    return c2w