"""
Converts rendered replica scenes from https://github.com/Harry-Zhi/semantic_nerf
to the autolabel scene format.

usage:
    python scripts/convert_replica.py <replica sequence> --out <output-directory>
"""
import argparse
import cv2
import json
import math
import numpy as np
import open3d as o3d
import os
import shutil
from tqdm import tqdm

from autolabel.utils import Scene, transform_points


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


class Exporter:

    def __init__(self, flags):
        self.flags = flags
        self.in_scene = flags.scene
        self._collect_paths()

    def _collect_paths(self):
        rgb_path = os.path.join(self.in_scene, 'rgb')
        depth_path = os.path.join(self.in_scene, 'depth')
        semantic_path = os.path.join(self.in_scene, 'semantic_class')
        rgb_frames = [f for f in os.listdir(rgb_path) if f[0] != '.']
        depth_frames = [f for f in os.listdir(depth_path) if f[0] != '.']
        semantic_frames = [
            f for f in os.listdir(semantic_path)
            if f[0] != '.' and 'semantic' in f
        ]
        rgb_frames = sorted(rgb_frames,
                            key=lambda x: int(x.split('_')[-1].split('.')[0]))
        depth_frames = sorted(depth_frames,
                              key=lambda x: int(x.split('_')[-1].split('.')[0]))
        semantic_frames = sorted(
            semantic_frames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.rgb_frames = []
        self.depth_frames = []
        self.semantic_frames = []
        for rgb, depth, semantic in zip(rgb_frames, depth_frames,
                                        semantic_frames):
            self.rgb_frames.append(os.path.join(rgb_path, rgb))
            self.depth_frames.append(os.path.join(depth_path, depth))
            self.semantic_frames.append(os.path.join(semantic_path, semantic))

    def _copy_frames(self):
        self.rgb_out = os.path.join(self.flags.out, 'rgb')
        self.depth_out = os.path.join(self.flags.out, 'depth')
        self.semantic_out = os.path.join(self.flags.out, 'semantic')
        os.makedirs(self.rgb_out, exist_ok=True)
        os.makedirs(self.depth_out, exist_ok=True)
        os.makedirs(self.semantic_out, exist_ok=True)

        semantic_classes = set()
        semantic_frames = []
        for i, (rgb, depth, semantic) in enumerate(
                zip(tqdm(self.rgb_frames, desc="Copying frames"),
                    self.depth_frames, self.semantic_frames)):
            rgb_out_path = os.path.join(self.rgb_out, f"{i:06}.png")
            depth_out_path = os.path.join(self.depth_out, f"{i:06}.png")
            shutil.copy(rgb, rgb_out_path)
            shutil.copy(depth, depth_out_path)

            sem_frame = cv2.imread(semantic, -1)
            semantic_frames.append(sem_frame)
            classes = np.unique(sem_frame)
            semantic_classes = semantic_classes.union(classes)

        for i, (frame, path) in enumerate(
                zip(tqdm(semantic_frames, desc="Writing semantic"),
                    self.semantic_frames)):
            new_semantic_seg = np.zeros_like(frame)
            for new_class_id, class_id in enumerate(semantic_classes):
                new_semantic_seg[frame == class_id] = new_class_id
            semantic_out = os.path.join(self.semantic_out, f"{i:06}.png")
            cv2.imwrite(semantic_out, new_semantic_seg)
        metadata = {'n_classes': len(semantic_classes)}
        metadata_path = os.path.join(self.flags.out, 'metadata.json')
        with open(metadata_path, 'w') as f:
            f.write(json.dumps(metadata, indent=2))

    def _copy_trajectory(self):
        pose_dir = os.path.join(self.flags.out, 'pose')
        os.makedirs(pose_dir, exist_ok=True)
        trajectory = np.loadtxt(os.path.join(self.flags.scene, 'traj_w_c.txt'),
                                delimiter=' ').reshape(-1, 4, 4)
        for i, T_CW in enumerate(trajectory):
            pose_out = os.path.join(pose_dir, f"{i:06}.txt")
            np.savetxt(pose_out, np.linalg.inv(T_CW))

    def _copy_intrinsics(self):
        width = 640
        height = 480
        hfov = 90.0
        fx = width / 2.0 / math.tan(math.radians(hfov / 2.0))
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0
        camera_matrix = np.eye(3)
        camera_matrix[0, 0] = fx
        camera_matrix[1, 1] = fx
        camera_matrix[0, 2] = cx
        camera_matrix[1, 2] = cy
        np.savetxt(os.path.join(self.flags.out, 'intrinsics.txt'),
                   camera_matrix)

    def _compute_bounds(self):
        scene = Scene(self.flags.out)
        depth_frame = o3d.io.read_image(scene.depth_paths()[0])
        depth_size = np.asarray(depth_frame).shape[::-1]
        K = scene.camera.scale(depth_size).camera_matrix
        intrinsics = o3d.camera.PinholeCameraIntrinsic(int(depth_size[0]),
                                                       int(depth_size[1]),
                                                       K[0, 0], K[1, 1],
                                                       K[0, 2], K[1, 2])
        pc = o3d.geometry.PointCloud()

        poses = scene.poses[::10]
        depths = scene.depth_paths()[::10]
        for T_CW, depth in zip(poses, tqdm(depths, desc="Computing bounds")):
            T_WC = np.linalg.inv(T_CW)
            depth = o3d.io.read_image(depth)

            pc_C = o3d.geometry.PointCloud.create_from_depth_image(
                depth, depth_scale=1000.0, intrinsic=intrinsics)
            pc_C = np.asarray(pc_C.points)
            pc_W = transform_points(T_WC, pc_C)

            pc += o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(pc_W)).uniform_down_sample(50)
        filtered, _ = pc.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
        aabb = filtered.get_axis_aligned_bounding_box()
        with open(os.path.join(scene.path, 'bbox.txt'), 'wt') as f:
            min_str = " ".join([str(x) for x in aabb.get_min_bound()])
            max_str = " ".join([str(x) for x in aabb.get_max_bound()])
            f.write(f"{min_str} {max_str} 0.01")

    def run(self):
        self._copy_frames()
        self._copy_trajectory()
        self._copy_intrinsics()
        self._compute_bounds()


if __name__ == "__main__":
    flags = read_args()
    Exporter(flags).run()
