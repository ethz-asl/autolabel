import argparse
import os
from os import path
import pickle
import queue
import cv2
import numpy as np
import open3d as o3d
from stray.scene import Scene
from scipy.spatial.transform import Rotation
from stray import linalg

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--visualize', '-v', action='store_true', help="Visualize for debugging.")
    return parser.parse_args()

def get_bounding_box(scene, poses):
    # Compute axis-aligned bounding box of the depth values in world frame.
    # Then get the center.
    min_bounds = np.zeros(3)
    max_bounds = np.zeros(3)
    pc = o3d.geometry.PointCloud()
    depth_frame = o3d.io.read_image(scene.get_depth_filepaths()[0])
    depth_size = np.asarray(depth_frame).shape[::-1]
    K = scene.camera().scale(depth_size).camera_matrix
    intrinsics = o3d.camera.PinholeCameraIntrinsic(int(depth_size[0]), int(depth_size[1]), K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    pc = o3d.geometry.PointCloud()
    for T_WC, depth in zip(scene.poses, scene.get_depth_filepaths()):
        depth = o3d.io.read_image(depth)
        pc_C = o3d.geometry.PointCloud.create_from_depth_image(depth, depth_scale=1000.0, intrinsic=intrinsics)
        pc_C = np.asarray(pc_C.points)
        pc_W = linalg.transform_points(T_WC, pc_C)
        min_bounds = np.minimum(min_bounds, pc_W.min(axis=0))
        max_bounds = np.maximum(max_bounds, pc_W.max(axis=0))
        pc += o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_W)).uniform_down_sample(50)

    filtered, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    bbox = filtered.get_oriented_bounding_box(robust=True)
    T = np.eye(4)
    T[:3, :3] = bbox.R.T
    o3d_aabb = o3d.geometry.PointCloud(filtered).transform(T).get_axis_aligned_bounding_box()
    center = o3d_aabb.get_center()
    T[:3, 3] = -center
    aabb = np.zeros((2, 3))
    aabb[0, :] = o3d_aabb.get_min_bound() - center
    aabb[1, :] = o3d_aabb.get_max_bound() - center
    return T, aabb, filtered

def main():
    flags = read_args()
    scene = Scene(flags.scene)

    poses = np.stack(scene.poses)
    T, aabb, pc = get_bounding_box(scene, poses)
    import ipdb; ipdb.set_trace()

    if flags.visualize:
        from stray.debugger import VisualDebugger
        poses = [T @ T_WC for T_WC in scene.poses]
        debugger = VisualDebugger()
        for T_WC in poses:
            debugger.add_frame(T_WC)
        debugger.add_mesh(pc.transform(T))
        debugger.add_frame(np.eye(4), scale=0.5)

        debugger.show()


    metadata = {
        'transform': T,
        'aabb': aabb
    }
    with open(os.path.join(flags.scene, 'nerf_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)


if __name__ == "__main__":
    main()