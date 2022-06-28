import os
import cv2
import numpy as np
from torch_ngp.nerf.network_ff import NeRFNetwork

class Camera:
    def __init__(self, camera_matrix, size):
        self.camera_matrix = camera_matrix
        self.size = size

    def scale(self, new_size):
        scale_x = new_size[0] / self.size[0]
        scale_y = new_size[1] / self.size[1]
        camera_matrix = self.camera_matrix.copy()
        camera_matrix[0, :] = scale_x * self.camera_matrix[0, :]
        camera_matrix[1, :] = scale_y * self.camera_matrix[1, :]
        return Camera(camera_matrix, new_size)

    @property
    def fx(self):
        return self.camera_matrix[0, 0]

    @property
    def fy(self):
        return self.camera_matrix[1, 1]

    @property
    def cx(self):
        return self.camera_matrix[0, 2]

    @property
    def cy(self):
        return self.camera_matrix[1, 2]

    @classmethod
    def from_path(self, path, size):
        return Camera(np.loadtxt(path), size)

class Scene:
    def __init__(self, scene_path):
        self.path = scene_path
        self.rgb_path = os.path.join(scene_path, 'rgb')
        self.depth_path = os.path.join(scene_path, 'depth')
        self.pose_path = os.path.join(scene_path, 'pose')
        self._read_poses()
        intrinsics_path = os.path.join(scene_path, 'intrinsics.txt')
        image_size = self._peak_image_size()
        self.camera = Camera.from_path(intrinsics_path, image_size)

    def _peak_image_size(self):
        image = cv2.imread(os.path.join(self.rgb_path, os.listdir(self.rgb_path)[0]))
        return (image.shape[1], image.shape[0])

    def _read_poses(self):
        pose_files = os.listdir(self.pose_path)
        pose_files = sorted([p for p in pose_files if p[0] != '.'], key=lambda p: int(p.split('.')[0]))
        self.poses = []
        for pose_file in pose_files:
            T_CW = np.loadtxt(os.path.join(self.pose_path, pose_file))
            self.poses.append(T_CW)

    def __iter__(self):
        rgb_frames = self.rgb_paths()
        depth_frames = self.depth_paths()
        for pose, rgb, depth in zip(self.poses, rgb_frames, depth_frames):
            yield (pose, rgb, depth)

    def __len__(self):
        return len(self.poses)

    def rgb_paths(self):
        rgb_frames = os.listdir(self.rgb_path)
        rgb_frames = sorted(rgb_frames, key=lambda x: int(x.split('.')[0]))
        return [os.path.join(self.rgb_path, f) for f in rgb_frames]

    def depth_paths(self):
        depth_frames = os.listdir(self.depth_path)
        depth_frames = sorted(depth_frames, key=lambda x: int(x.split('.')[0]))
        return [os.path.join(self.depth_path, d) for d in depth_frames]

    def bbox(self):
        return np.loadtxt(os.path.join(self.path, 'bbox.txt'))[:6]

def transform_points(T, points):
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points[..., :, None])[..., :, 0] + t

def create_model(dataset):
    extents = dataset.max_bounds - dataset.min_bounds
    bound = (extents - (dataset.min_bounds + dataset.max_bounds) * 0.5).max()
    return NeRFNetwork(num_layers=2, num_layers_color=2,
            hidden_dim_color=64,
            hidden_dim=64,
            geo_feat_dim=15,
            encoding="hashgrid",
            bound=float(bound),
            cuda_ray=False,
            density_scale=1)

