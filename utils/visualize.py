import cv2
import numpy as np
import os
import os.path as osp
from plyfile import PlyData, PlyElement


class MMeshMeta(object):
    def __init__(self) -> None:
        super(MMeshMeta, self).__init__()
        self.vertices = []
        self.faces = []

    def __repr__(self) -> str:
        s = f"[MeshMeta] vertices({len(self.vertices)}), faces({len(self.faces.shape)})."
        return s

    def __str__(self) -> str:
        return self.__repr__()

    def parse_aabb(self, aabb):
        if isinstance(aabb, (list, tuple)):
            aabb_ = np.array(aabb)
        elif isinstance(aabb, np.ndarray):
            aabb_ = aabb
        else:
            raise TypeError(f"aabb type is {type(aabb)}, not (List[int], Tuple[int], np.ndarray).")
        assert aabb_.shape == (2, 3), f"aabb shape is {aabb_.shape}, not (2, 3)."
        x_min, y_min, z_min = aabb_[0].tolist()
        x_max, y_max, z_max = aabb_[1].tolist()
        return x_min, y_min, z_min, x_max, y_max, z_max

    def parse_bbox(self, bbox):
        """_summary_

        Args:
            bbox (_type_): 8 points, [x, y, z]

        Raises:
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        if isinstance(bbox, (list, tuple)):
            bbox_ = np.array(bbox)
        elif isinstance(bbox, np.ndarray):
            bbox_ = bbox
        else:
            raise TypeError(f"bbox type is {type(bbox)}, not (List[int], Tuple[int], np.ndarray).")
        assert bbox_.shape == (8, 3), f"bbox shape is {bbox_.shape}, not (8, 3)."
        p0 = bbox_[0].tolist()
        p1 = bbox_[1].tolist()
        p2 = bbox_[2].tolist()
        p3 = bbox_[3].tolist()
        p4 = bbox_[4].tolist()
        p5 = bbox_[5].tolist()
        p6 = bbox_[6].tolist()
        p7 = bbox_[7].tolist()
        return p0, p1, p2, p3, p4, p5, p6, p7

    def parse_color(self, color):
        """_summary_

        Args:
            color (_type_): 0-255 range, [r, g, b]

        Raises:
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        if isinstance(color, (list, tuple)):
            color_ = list(color)
        elif isinstance(color, np.ndarray):
            color_ = color.tolist()
        else:
            raise TypeError(f"color type is {type(color)}, not (List[int], Tuple[int], np.ndarray).")
        assert len(color_) == 3, f"color shape is {len(color_)}, not (3)."  # [r, g, b]
        return color_

    def parse_colors(self, colors):
        """_summary_

        Args:
            color (_type_): 0-255 range, [r, g, b]

        Raises:
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        if isinstance(colors, (list, tuple)):
            colors_ = np.array(colors)
        elif isinstance(colors, np.ndarray):
            colors_ = colors
        else:
            raise TypeError(f"colors type is {type(colors)}, not (List[int], Tuple[int], np.ndarray).")
        assert colors_.shape[1] == 3, f"points shape is {colors_.shape}, not (n, 3)."  # [r, g, b]
        return colors_.tolist()

    def parse_point(self, point):
        if isinstance(point, (list, tuple)):
            point_ = list(point)
        elif isinstance(point, np.ndarray):
            point_ = point.tolist()
        else:
            raise TypeError(f"point type is {type(point)}, not (List[int], Tuple[int], np.ndarray).")
        assert len(point_) == 3, f"point shape is {len(point_)}, not (3)."
        return point_

    def parse_points(self, points):
        if isinstance(points, (list, tuple)):
            points_ = np.array(points)
        elif isinstance(points, np.ndarray):
            points_ = points
        else:
            raise TypeError(f"points type is {type(points)}, not (List[int], Tuple[int], np.ndarray).")
        assert points_.shape[1] == 3, f"points shape is {points_.shape}, not (n, 3)."
        return points_.tolist()

    def create_aabb_mesh(self, aabb, color):
        x_min, y_min, z_min, x_max, y_max, z_max = self.parse_aabb(aabb)
        bbox = np.array([[x_max, y_min, z_min], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_max, y_max, z_min],
                         [x_min, y_min, z_min], [x_min, y_min, z_max], [x_min, y_max, z_max], [x_min, y_max, z_min]])
        return self.create_bbox_mesh(bbox, color)

    def create_bbox_mesh(self, bbox, color):
        """_summary_

        Args:
            bbox (_type_): clockwise & top to down, [x, y, z]
            color (_type_): 0-255 range, [r, g, b]
        """
        p0, p1, p2, p3, p4, p5, p6, p7 = self.parse_bbox(bbox)
        color_ = self.parse_color(color)
        self.create_prism4_mesh(np.array([p0, p1, p2, p3]), np.array([p4, p5, p6, p7]), color_)
        return True

    def create_pyramid3_mesh(self, sp, ef, color):
        # triangular pyramid, clockwise & top to down
        sp_ = self.parse_point(sp)
        ef_ = self.parse_points(ef)
        assert len(ef_) == 3, f"point number of ef is {len(ef_)}, not (3)."
        color_ = self.parse_color(color)
        vertex_start = len(self.vertices)
        self.vertices += [tuple(sp_ + color_), tuple(ef_[0] + color_), tuple(ef_[1] + color_), tuple(ef_[2] + color_)]
        self.faces += [
            tuple([[vertex_start + 0, vertex_start + 2, vertex_start + 1]] + color_),  # side
            tuple([[vertex_start + 0, vertex_start + 3, vertex_start + 2]] + color_),
            tuple([[vertex_start + 0, vertex_start + 1, vertex_start + 3]] + color_),
            tuple([[vertex_start + 1, vertex_start + 2, vertex_start + 3]] + color_)  # bottom
        ]
        return True

    def create_pyramid4_mesh(self, sp, ef, color):
        # quadrilateral pyramid, clockwise & top to down
        sp_ = self.parse_point(sp)
        ef_ = self.parse_points(ef)
        assert len(ef_) == 4, f"point number of ef is {len(ef_)}, not (4)."
        color_ = self.parse_color(color)
        vertex_start = len(self.vertices)
        self.vertices += [
            tuple(sp_ + color_),
            tuple(ef_[0] + color_),
            tuple(ef_[1] + color_),
            tuple(ef_[2] + color_),
            tuple(ef_[3] + color_)
        ]
        self.faces += [
            tuple([[vertex_start + 0, vertex_start + 2, vertex_start + 1]] + color_),
            tuple([[vertex_start + 0, vertex_start + 3, vertex_start + 2]] + color_),
            tuple([[vertex_start + 0, vertex_start + 4, vertex_start + 3]] + color_),
            tuple([[vertex_start + 0, vertex_start + 1, vertex_start + 4]] + color_),
            tuple([[vertex_start + 1, vertex_start + 2, vertex_start + 3]] + color_),
            tuple([[vertex_start + 3, vertex_start + 4, vertex_start + 1]] + color_)
        ]
        return True

    def create_prism3_mesh(self, sf, ef, color):
        # triangular prism, clockwise & top to down
        sf_ = self.parse_points(sf)
        assert len(sf_) == 3, f"point number of sf is {len(sf_)}, not (3)."
        ef_ = self.parse_points(ef)
        assert len(ef_) == 3, f"point number of ef is {len(ef_)}, not (3)."
        color_ = self.parse_color(color)
        vertex_start = len(self.vertices)
        self.vertices += [
            tuple(sf_[0] + color_),
            tuple(sf_[1] + color_),
            tuple(sf_[2] + color_),
            tuple(ef_[0] + color_),
            tuple(ef_[1] + color_),
            tuple(ef_[2] + color_)
        ]
        self.faces += [
            tuple([[vertex_start + 0, vertex_start + 2, vertex_start + 1]] + color_),  # top
            tuple([[vertex_start + 0, vertex_start + 1, vertex_start + 4]] + color_),  # side
            tuple([[vertex_start + 4, vertex_start + 3, vertex_start + 0]] + color_),
            tuple([[vertex_start + 1, vertex_start + 2, vertex_start + 5]] + color_),
            tuple([[vertex_start + 5, vertex_start + 4, vertex_start + 1]] + color_),
            tuple([[vertex_start + 2, vertex_start + 0, vertex_start + 3]] + color_),
            tuple([[vertex_start + 3, vertex_start + 5, vertex_start + 2]] + color_),
            tuple([[vertex_start + 3, vertex_start + 4, vertex_start + 5]] + color_)  # bottom
        ]
        return True

    def create_prism4_mesh(self, sf, ef, color):
        # quadrilateral prism, clockwise & top to down
        sf_ = self.parse_points(sf)
        assert len(sf_) == 4, f"point number of sf is {len(sf_)}, not (4)."
        ef_ = self.parse_points(ef)
        assert len(ef_) == 4, f"point number of ef is {len(ef_)}, not (4)."
        color_ = self.parse_color(color)
        vertex_start = len(self.vertices)
        self.vertices += [
            tuple(sf_[0] + color_),
            tuple(sf_[1] + color_),
            tuple(sf_[2] + color_),
            tuple(sf_[3] + color_),
            tuple(ef_[0] + color_),
            tuple(ef_[1] + color_),
            tuple(ef_[2] + color_),
            tuple(ef_[3] + color_)
        ]
        self.faces += [
            tuple([[vertex_start + 0, vertex_start + 2, vertex_start + 1]] + color_),  # top
            tuple([[vertex_start + 0, vertex_start + 3, vertex_start + 2]] + color_),
            tuple([[vertex_start + 0, vertex_start + 1, vertex_start + 4]] + color_),  # side
            tuple([[vertex_start + 1, vertex_start + 5, vertex_start + 4]] + color_),
            tuple([[vertex_start + 1, vertex_start + 2, vertex_start + 5]] + color_),
            tuple([[vertex_start + 2, vertex_start + 6, vertex_start + 5]] + color_),
            tuple([[vertex_start + 2, vertex_start + 3, vertex_start + 6]] + color_),
            tuple([[vertex_start + 3, vertex_start + 7, vertex_start + 6]] + color_),
            tuple([[vertex_start + 3, vertex_start + 0, vertex_start + 7]] + color_),
            tuple([[vertex_start + 0, vertex_start + 4, vertex_start + 7]] + color_),
            tuple([[vertex_start + 4, vertex_start + 5, vertex_start + 6]] + color_),  # bottom
            tuple([[vertex_start + 4, vertex_start + 6, vertex_start + 7]] + color_)
        ]
        return True

    def create_line_mesh(self, sp, ep, color, eps=1.0e-2):
        sp_ = self.parse_point(sp)
        sp0 = [sp_[0] + eps, sp_[1], sp_[2]]
        sp1 = [sp_[0], sp_[1] + eps, sp_[2]]
        sp2 = [sp_[0], sp_[1], sp_[2] + eps]
        ep_ = self.parse_point(ep)
        ep0 = [ep_[0] + eps, ep_[1], ep_[2]]
        ep1 = [ep_[0], ep_[1] + eps, ep_[2]]
        ep2 = [ep_[0], ep_[1], ep_[2] + eps]
        color_ = self.parse_color(color)
        vertex_start = len(self.vertices)
        self.vertices += [
            tuple(sp0 + color_),
            tuple(sp1 + color_),
            tuple(sp2 + color_),
            tuple(ep0 + color_),
            tuple(ep1 + color_),
            tuple(ep2 + color_)
        ]
        self.faces += [
            tuple([[vertex_start + 0, vertex_start + 1, vertex_start + 4]] + color_),
            tuple([[vertex_start + 4, vertex_start + 3, vertex_start + 0]] + color_),
            tuple([[vertex_start + 1, vertex_start + 2, vertex_start + 5]] + color_),
            tuple([[vertex_start + 5, vertex_start + 4, vertex_start + 1]] + color_),
            tuple([[vertex_start + 2, vertex_start + 0, vertex_start + 3]] + color_),
            tuple([[vertex_start + 3, vertex_start + 5, vertex_start + 2]] + color_),
            tuple([[vertex_start + 0, vertex_start + 2, vertex_start + 1]] + color_),
            tuple([[vertex_start + 3, vertex_start + 4, vertex_start + 5]] + color_),
        ]
        return True

    def create_aabb_skeleton(self, aabb, color, eps=1.0e-2):
        x_min, y_min, z_min, x_max, y_max, z_max = self.parse_aabb(aabb)
        bbox = np.array([[x_max, y_min, z_min], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_max, y_max, z_min],
                         [x_min, y_min, z_min], [x_min, y_min, z_max], [x_min, y_max, z_max], [x_min, y_max, z_min]])
        return self.create_bbox_skeleton(bbox, color, eps)

    def create_bbox_skeleton(self, bbox, color, eps=1.0e-2):
        p0, p1, p2, p3, p4, p5, p6, p7 = self.parse_bbox(bbox)
        color_ = self.parse_color(color)
        self.create_line_mesh(p0, p1, color_, eps=eps)  # top
        self.create_line_mesh(p1, p2, color_, eps=eps)
        self.create_line_mesh(p3, p2, color_, eps=eps)
        self.create_line_mesh(p0, p3, color_, eps=eps)
        self.create_line_mesh(p4, p0, color_, eps=eps)  # side
        self.create_line_mesh(p5, p1, color_, eps=eps)
        self.create_line_mesh(p6, p2, color_, eps=eps)
        self.create_line_mesh(p7, p3, color_, eps=eps)
        self.create_line_mesh(p4, p5, color_, eps=eps)  # bottom
        self.create_line_mesh(p5, p6, color_, eps=eps)
        self.create_line_mesh(p7, p6, color_, eps=eps)
        self.create_line_mesh(p4, p7, color_, eps=eps)
        return True

    def create_pyramid3_skeleton(self, sp, ef, color, eps=1.0e-2):
        # triangular pyramid, clockwise & top to down
        sp_ = self.parse_point(sp)
        ef_ = self.parse_points(ef)
        assert len(ef_) == 3, f"point number of ef is {len(ef_)}, not (3)."
        color_ = self.parse_color(color)
        self.create_line_mesh(sp_, ef_[0], color_, eps=eps)  # side
        self.create_line_mesh(sp_, ef_[1], color_, eps=eps)
        self.create_line_mesh(sp_, ef_[2], color_, eps=eps)
        self.create_line_mesh(ef_[0], ef_[1], color_, eps=eps)  # bottom
        self.create_line_mesh(ef_[1], ef_[2], color_, eps=eps)
        self.create_line_mesh(ef_[2], ef_[0], color_, eps=eps)
        return True

    def create_pyramid4_skeleton(self, sp, ef, color, eps=1.0e-2):
        # quadrilateral pyramid, clockwise & top to down
        sp_ = self.parse_point(sp)
        ef_ = self.parse_points(ef)
        assert len(ef_) == 4, f"point number of ef is {len(ef_)}, not (4)."
        color_ = self.parse_color(color)
        self.create_line_mesh(sp_, ef_[0], color_, eps=eps)  # side
        self.create_line_mesh(sp_, ef_[1], color_, eps=eps)
        self.create_line_mesh(sp_, ef_[2], color_, eps=eps)
        self.create_line_mesh(sp_, ef_[3], color_, eps=eps)
        self.create_line_mesh(ef_[0], ef_[1], color_, eps=eps)  # bottom
        self.create_line_mesh(ef_[1], ef_[2], color_, eps=eps)
        self.create_line_mesh(ef_[3], ef_[2], color_, eps=eps)
        self.create_line_mesh(ef_[0], ef_[3], color_, eps=eps)
        return True

    def create_prism3_skeleton(self, sf, ef, color, eps=1.0e-2):
        # triangular prism, clockwise & top to down
        sf_ = self.parse_points(sf)
        assert len(sf_) == 3, f"point number of sf is {len(sf_)}, not (3)."
        ef_ = self.parse_points(ef)
        assert len(ef_) == 3, f"point number of ef is {len(ef_)}, not (3)."
        color_ = self.parse_color(color)
        self.create_line_mesh(sf_[0], sf_[1], color_, eps=eps)  # top
        self.create_line_mesh(sf_[1], sf_[2], color_, eps=eps)
        self.create_line_mesh(sf_[0], sf_[2], color_, eps=eps)
        self.create_line_mesh(sf_[0], ef_[0], color_, eps=eps)  # side
        self.create_line_mesh(sf_[1], ef_[1], color_, eps=eps)
        self.create_line_mesh(sf_[2], ef_[2], color_, eps=eps)
        self.create_line_mesh(ef_[0], ef_[1], color_, eps=eps)  # bottom
        self.create_line_mesh(ef_[1], ef_[2], color_, eps=eps)
        self.create_line_mesh(ef_[0], ef_[2], color_, eps=eps)
        return True

    def create_prism4_skeleton(self, sf, ef, color, eps=1.0e-2):
        # quadrilateral prism, clockwise & top to down
        sf_ = self.parse_points(sf)
        assert len(sf_) == 4, f"point number of sf is {len(sf_)}, not (4)."
        ef_ = self.parse_points(ef)
        assert len(ef_) == 4, f"point number of ef is {len(ef_)}, not (4)."
        color_ = self.parse_color(color)
        self.create_line_mesh(sf_[0], sf_[1], color_, eps=eps)  # top
        self.create_line_mesh(sf_[1], sf_[2], color_, eps=eps)
        self.create_line_mesh(sf_[3], sf_[2], color_, eps=eps)
        self.create_line_mesh(sf_[0], sf_[3], color_, eps=eps)
        self.create_line_mesh(sf_[0], ef_[0], color_, eps=eps)  # side
        self.create_line_mesh(sf_[1], ef_[1], color_, eps=eps)
        self.create_line_mesh(sf_[2], ef_[2], color_, eps=eps)
        self.create_line_mesh(sf_[3], ef_[3], color_, eps=eps)
        self.create_line_mesh(ef_[0], ef_[1], color_, eps=eps)  # bottom
        self.create_line_mesh(ef_[1], ef_[2], color_, eps=eps)
        self.create_line_mesh(ef_[3], ef_[2], color_, eps=eps)
        self.create_line_mesh(ef_[0], ef_[3], color_, eps=eps)
        return True

    def create_frame_coordinates(self, length=1., eps=1.0e-2):
        self.create_line_mesh([0., 0., 0.], [length, 0., 0.], [255, 0, 0], eps=eps)  # x
        self.create_line_mesh([0., 0., 0.], [0., length, 0.], [0, 255, 0], eps=eps)  # y
        self.create_line_mesh([0., 0., 0.], [0., 0., length], [0, 0, 255], eps=eps)  # z
        return True

    def create_camera_coordinates(self, exmat, length=1., eps=1.0e-2):
        # single camera coordinate, exmat is extrinsic matrix [4, 4]
        p = exmat[:3, 3]
        px = p + length * exmat[:3, 0]
        py = p + length * exmat[:3, 1]
        pz = p + length * exmat[:3, 2]
        self.create_line_mesh(p, px, [255, 0, 0], eps=eps)  # x
        self.create_line_mesh(p, py, [0, 255, 0], eps=eps)  # y
        self.create_line_mesh(p, pz, [0, 0, 255], eps=eps)  # z
        return True

    def create_camera_extrinsic(self, exmat, inmat, color, length=1., eps=1.0e-2):
        # single camera camera extrinsic (pyramid), exmat is extrinsic matrix [4, 4]
        # inmat is intrinsic matrix [3, 3]
        p = exmat[:3, 3]
        fx, fy, cx, cy = inmat[0, 0], inmat[1, 1], inmat[0, 2], inmat[1, 2]
        p_ = p + length * exmat[:3, 2]  # center, assert camera is looking at z axis
        tl = p_ + length * (exmat[:3, 0] * (-cx / fx) + exmat[:3, 1] * (-cy / fy))
        tr = p_ + length * (exmat[:3, 0] * (cx / fx) + exmat[:3, 1] * (-cy / fy))
        bl = p_ + length * (exmat[:3, 0] * (-cx / fx) + exmat[:3, 1] * (cy / fy))
        br = p_ + length * (exmat[:3, 0] * (cx / fx) + exmat[:3, 1] * (cy / fy))
        self.create_pyramid4_skeleton(p, [tl, tr, br, bl], color, eps=eps)
        return True

    def create_camera_bounds_mesh(self, exmat, inmat, bds, near_color, far_color, eps=1.0e-2):
        # single camera camera extrinsic (pyramid), exmat is extrinsic matrix [4, 4]
        # inmat is intrinsic matrix [3, 3], bds is bounds [2]
        p = exmat[:3, 3]
        fx, fy, cx, cy = inmat[0, 0], inmat[1, 1], inmat[0, 2], inmat[1, 2]
        near, far = bds[0], bds[1]
        p_near = p + near * exmat[:3, 2]  # center, assert camera is looking at z axis
        tl_near = p_near + near * (exmat[:3, 0] * (-cx / fx) + exmat[:3, 1] * (-cy / fy))
        tr_near = p_near + near * (exmat[:3, 0] * (cx / fx) + exmat[:3, 1] * (-cy / fy))
        bl_near = p_near + near * (exmat[:3, 0] * (-cx / fx) + exmat[:3, 1] * (cy / fy))
        br_near = p_near + near * (exmat[:3, 0] * (cx / fx) + exmat[:3, 1] * (cy / fy))
        p_far = p + far * exmat[:3, 2]  # center, assert camera is looking at z axis
        tl_far = p_far + far * (exmat[:3, 0] * (-cx / fx) + exmat[:3, 1] * (-cy / fy))
        tr_far = p_far + far * (exmat[:3, 0] * (cx / fx) + exmat[:3, 1] * (-cy / fy))
        bl_far = p_far + far * (exmat[:3, 0] * (-cx / fx) + exmat[:3, 1] * (cy / fy))
        br_far = p_far + far * (exmat[:3, 0] * (cx / fx) + exmat[:3, 1] * (cy / fy))
        self.create_pyramid4_skeleton(p, [tl_near, tr_near, br_near, bl_near], near_color, eps=eps)
        self.create_prism4_mesh([tl_near, tr_near, br_near, bl_near], [tl_far, tr_far, br_far, bl_far], far_color)
        return True

    def create_camera_bounds_skeleton(self, exmat, inmat, bds, near_color, far_color, eps=1.0e-2):
        # single camera camera extrinsic (pyramid), exmat is extrinsic matrix [4, 4]
        # inmat is intrinsic matrix [3, 3], bds is bounds [2]
        p = exmat[:3, 3]
        fx, fy, cx, cy = inmat[0, 0], inmat[1, 1], inmat[0, 2], inmat[1, 2]
        near, far = bds[0], bds[1]
        p_near = p + near * exmat[:3, 2]  # center, assert camera is looking at z axis
        tl_near = p_near + near * (exmat[:3, 0] * (-cx / fx) + exmat[:3, 1] * (-cy / fy))
        tr_near = p_near + near * (exmat[:3, 0] * (cx / fx) + exmat[:3, 1] * (-cy / fy))
        bl_near = p_near + near * (exmat[:3, 0] * (-cx / fx) + exmat[:3, 1] * (cy / fy))
        br_near = p_near + near * (exmat[:3, 0] * (cx / fx) + exmat[:3, 1] * (cy / fy))
        p_far = p + far * exmat[:3, 2]  # center, assert camera is looking at z axis
        tl_far = p_far + far * (exmat[:3, 0] * (-cx / fx) + exmat[:3, 1] * (-cy / fy))
        tr_far = p_far + far * (exmat[:3, 0] * (cx / fx) + exmat[:3, 1] * (-cy / fy))
        bl_far = p_far + far * (exmat[:3, 0] * (-cx / fx) + exmat[:3, 1] * (cy / fy))
        br_far = p_far + far * (exmat[:3, 0] * (cx / fx) + exmat[:3, 1] * (cy / fy))
        self.create_pyramid4_skeleton(p, [tl_near, tr_near, br_near, bl_near], near_color, eps=eps)
        self.create_prism4_skeleton([tl_near, tr_near, br_near, bl_near], [tl_far, tr_far, br_far, bl_far],
                                    far_color,
                                    eps=eps)
        return True

    def create_pointcloud(self, points, colors):
        points_ = self.parse_points(points)
        clolors_ = self.parse_colors(colors)
        assert len(points_) == len(clolors_), "points and colors should have same length, " \
            f"but got {len(points_)} and {len(clolors_)}"
        for p, c in zip(points_, clolors_):
            self.vertices += [tuple(p + c)]
        return True

    def save(self, path):
        assert path.endswith('.ply'), f"{path} is not a ply file! Only support ply file."
        if osp.exists(path):
            os.remove(path)
        vertex = np.array(self.vertices,
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        vertex_el = PlyElement.describe(vertex, 'vertex')
        face = np.array(self.faces,
                        dtype=[('vertex_indices', 'i4', (3, )), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        face_el = PlyElement.describe(face, 'face')
        PlyData([vertex_el, face_el]).write(path)
        return True

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('z')
        self.ax.set_ylabel('x')
        self.ax.set_zlabel('y')
        self.ax.scatter(0, 0, 0, c='k', marker='o')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color='r', focal_len_scaled=5, aspect_ratio=0.3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]
        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title(f'Sample')
        plt.show()

def resize_propper(image, max_size = 512):
    scale = 512 / np.max(image.shape[:2])
    image = cv2.resize(image, (0,0), fx=scale, fy=scale)