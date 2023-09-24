import json
import numpy as np
import matplotlib.pyplot as plt

json_file_path = './temp/sample.json'
colors = ('r', 'g', 'b', 'k')
ax = plt.figure().add_subplot(projection='3d')


def get_cam_sperical_coord(img_meta):
    c2w = np.array(img_meta["head"]["00"]["camera"][:16]).reshape(4, 4)
    T = c2w[:3, 3]
    x, y, z = T
    theta = np.rad2deg(np.arctan(np.sqrt(x**2+z**2)/y))
    phi = np.rad2deg(np.arctan(z/x))+180
    return theta, phi, x, y, z

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