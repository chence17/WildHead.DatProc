'''
Author: chence antonio.chan.cc@outlook.com
Date: 2023-09-24 16:32:08
LastEditors: chence antonio.chan.cc@outlook.com
LastEditTime: 2023-09-24 19:09:48
FilePath: /DatProc/X3.vis_angle.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import numpy as np
import matplotlib.pyplot as plt

from utils.visualize import CameraPoseVisualizer

json_file_path = './temp/sample.json'
colors = ('r', 'g', 'b', 'k')


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

vis = CameraPoseVisualizer([-2, 2], [-2, 2], [-2, 2])
for delta_theta in range(-180, 181, 15):
    vis.extrinsic2pyramid(gen_cam_matrix(delta_theta, 90, 1), 'b', 0.5)
for delta_phi in range(-180, 181, 15):
    vis.extrinsic2pyramid(gen_cam_matrix(90, delta_phi, 1), 'g', 0.5)
vis.show()




# with open(json_file_path, 'r') as f:
#     data = json.load(f)
#     img_meta = data["Data/000101.png"]["head"]["00"]["camera"]
#     c2w = np.array(img_meta).reshape(5,5)
#     theta, phi, r, x, y, z = get_cam_coords(c2w)
#     ax.scatter(z, x, y, c='r', marker='o')
#     ax.scatter(0, 0, 0, c='k', marker='o')
#     ax.plot((0, z), (0, x), (0, y),  c='k', label='w2c')
#     w2c_vec = np.array([z, x, y])
#     cam_z = -w2c_vec/np.linalg.norm(w2c_vec)
#     down_vec = np.array([0, 0, -1])
#     cam_x = np.cross(down_vec, cam_z)
#     cam_x = cam_x/np.linalg.norm(cam_x)
#     cam_y = np.cross(cam_z, cam_x)
#     cam_y = cam_y/np.linalg.norm(cam_y)
#     ax.plot((z, z+cam_x[0]), (x, x+cam_x[1]), (y, y+cam_x[2]),  c='r', label='cam_x')
#     ax.plot((z, z+cam_y[0]), (x, x+cam_y[1]), (y, y+cam_y[2]),  c='g', label='cam_y')
#     ax.plot((z, z+cam_z[0]), (x, x+cam_z[1]), (y, y+cam_z[2]),  c='b', label='cam_z')
#     ax.legend()
# # # Fixing random state for reproducibility
# # ax.set_xlim(0, 1)
# # ax.set_ylim(0, 1)
# # ax.set_zlim(0, 1)
# ax.set_xlabel('Z')
# ax.set_ylabel('X')
# ax.set_zlabel('Y')
# ax.view_init(elev=20., azim=0, roll=0)

# plt.show()