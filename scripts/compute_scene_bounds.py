description = """
This script computes the scene bounding box file (<scene>/bbox.txt) from depth and color images and camera poses.

The --vis flag can be used to visualize a point cloud of the scene.

Usage:
    python scripts/compute_scene_bounds.py <scene> [--vis]
"""
import numpy as np
import os
import argparse
from argparse import RawTextHelpFormatter
import open3d as o3d
import cv2
from autolabel.utils import Scene

def read_args():
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('scene')
    parser.add_argument('--vis', action='store_true')
    return parser.parse_args()

class BBoxComputer:

    def __init__(self, K, image_size):
        self.min_bounds = np.zeros(3)
        self.max_bounds = np.zeros(3)
        self.K = o3d.camera.PinholeCameraIntrinsic(int(image_size[0]),
                                                   int(image_size[1]), K[0, 0],
                                                   K[1, 1], K[0, 2], K[1, 2])
        self.pc = o3d.geometry.PointCloud()

    def add_frame(self, T_CW, depth):
        depth = o3d.geometry.Image(depth)
        pc_C = o3d.geometry.PointCloud.create_from_depth_image(
            depth, depth_scale=1000.0, intrinsic=self.K)
        pc_C = np.asarray(pc_C.points)
        T_WC = np.linalg.inv(T_CW)
        pc_W = (T_WC[:3, :3].__matmul__(pc_C[:, :, None]))[:, :, 0] + T_WC[:3,
                                                                           3]

        self.min_bounds = np.minimum(self.min_bounds, pc_W.min(axis=0))
        self.max_bounds = np.maximum(self.max_bounds, pc_W.max(axis=0))
        self.pc += o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(pc_W)).uniform_down_sample(50)

    def get_bounds(self):
        filtered, _ = self.pc.remove_statistical_outlier(nb_neighbors=20,
                                                         std_ratio=2.0)
        o3d_aabb = o3d.geometry.PointCloud(filtered).get_axis_aligned_bounding_box()
        aabb = np.zeros((2, 3))
        aabb[0, :] = o3d_aabb.get_min_bound()
        aabb[1, :] = o3d_aabb.get_max_bound()
        return aabb

def to_pointcloud(color, depth, T_CW, camera_matrix, image_size):
    depth = o3d.geometry.Image(depth)
    color = o3d.geometry.Image(color)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,
                depth_scale=1000.0, convert_rgb_to_intensity=False)

    pinhole = o3d.camera.PinholeCameraIntrinsic(image_size[0], image_size[1],
                camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2])
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole, extrinsic=T_CW)


def main():
    flags = read_args()

    scene = Scene(flags.scene)
    image_size = scene.peak_image_size()
    bbox_computer = BBoxComputer(scene.camera.camera_matrix, image_size)
    geometry = []

    data = [i for i in zip(scene.depth_paths(), scene.rgb_paths(), scene.poses)]
    for depth_path, rgb_path, T_CW in data[::60]:
        depth = cv2.imread(depth_path, -1)
        bbox_computer.add_frame(T_CW, depth)

        if flags.vis:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(0.1, np.zeros(3))
            geometry.append(axis.transform(np.linalg.inv(T_CW)))

            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            pc = to_pointcloud(rgb, depth, T_CW, scene.camera.camera_matrix, image_size)
            geometry.append(pc)

    bounds = bbox_computer.get_bounds()
    bbox_path = os.path.join(flags.scene, 'bbox.txt')
    with open(bbox_path, 'wt') as f:
        min_str = " ".join([str(x) for x in bounds[0]])
        max_str = " ".join([str(x) for x in bounds[1]])
        f.write("{} {} 0.01".format(min_str, max_str))

    if flags.vis:
        o3d.visualization.draw_geometries(geometry)

if __name__ == "__main__":
    main()

