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

json_file_path = './temp/sample.json'
colors = ('r', 'g', 'b', 'k')
ax = plt.figure().add_subplot(projection='3d')


# def get_cam_coords(img_meta):
#     # World Coordinate System: x(right), y(up), z(forward)
#     c2w = np.array(img_meta["head"]["00"]["camera"][:16]).reshape(4, 4)
#     T = c2w[:3, 3]
#     x, y, z = T
#     r = np.sqrt(x**2+y**2+z**2)
#     # theta = np.rad2deg(np.arctan(np.sqrt(x**2+z**2)/y))
#     theta = np.arctan2(x, z)
#     # phi = np.rad2deg(np.arctan(z/x))+180
#     phi = np.arcsin(y/r)
#     return [theta, phi, r, x, y, z] # [:3] sperical cood, [3:] cartesian cood

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

with open(json_file_path, 'r') as f:
    data = json.load(f)

for img_meta in data.values():
    theta, phi, x, y, z = get_cam_sperical_coord(img_meta)
    ax.scatter(z, x, y, c='r', marker='o')
    ax.plot((0, z), (0, x), (0, y),  c='b')
# # Fixing random state for reproducibility
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
ax.set_xlabel('Z')
ax.set_ylabel('X')
ax.set_zlabel('Y')
# ax.view_init(elev=20., azim=-35, roll=0)

plt.savefig('./temp/X3.vis_angle.png')