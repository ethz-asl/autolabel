import cv2
import numpy as np
import os

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
        self.raw_rgb_path = os.path.join(scene_path, 'raw_rgb')
        self.depth_path = os.path.join(scene_path, 'depth')
        self.raw_depth_path = os.path.join(scene_path, 'raw_depth')
        self.pose_path = os.path.join(scene_path, 'pose')
        self._read_poses()
        intrinsics_path = os.path.join(scene_path, 'intrinsics.txt')
        image_size = self._peak_image_size()
        self.camera = Camera.from_path(intrinsics_path, image_size)

    def _peak_image_size(self):
        if os.path.exists(self.raw_rgb_path):
            path = self.raw_rgb_path
        elif os.path.exists(self.rgb_path):
            path = self.rgb_path
        else:
            raise ValueError("Doesn't appear to be a valid scene.")
        image = cv2.imread(os.path.join(path, os.listdir(path)[0]))
        return (image.shape[1], image.shape[0])

    def _read_poses(self):
        if not os.path.exists(self.pose_path):
            self.poses = []
            return
        pose_files = os.listdir(self.pose_path)
        pose_files = sorted([p for p in pose_files if p[0] != '.'],
                            key=lambda p: int(p.split('.')[0]))
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

    def _get_paths(self, directory):
        frames = os.listdir(directory)
        frames = sorted(frames, key=lambda x: int(x.split('.')[0]))
        return [os.path.join(directory, f) for f in frames]

    def rgb_paths(self):
        return self._get_paths(self.rgb_path)

    def depth_paths(self):
        return self._get_paths(self.depth_path)

    def raw_rgb_paths(self):
        return self._get_paths(self.raw_rgb_path)

    def raw_depth_paths(self):
        return self._get_paths(self.raw_depth_path)

    def image_names(self):
        """
        Returns the filenames of rgb images without file extensions.
        """
        rgb_frames = os.listdir(self.rgb_path)
        rgb_frames = sorted(rgb_frames, key=lambda x: int(x.split('.')[0]))
        return [f.split('.')[0] for f in rgb_frames]

    def bbox(self):
        return np.loadtxt(os.path.join(self.path, 'bbox.txt'))[:6]

    def depth_size(self):
        """
        Return: the size (width, height) of the depth images.
        """
        depth_paths = self.raw_depth_paths()
        if len(depth_paths) == 0:
            depth_paths = self.depth_paths()
        image = cv2.imread(depth_paths[0], -1)
        return (image.shape[1], image.shape[0])


def transform_points(T, points):
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points[..., :, None])[..., :, 0] + t


def create_model(dataset):
    extents = dataset.max_bounds - dataset.min_bounds
    bound = (extents - (dataset.min_bounds + dataset.max_bounds) * 0.5).max()
    return NeRFNetwork(num_layers=2,
                       num_layers_color=2,
                       hidden_dim_color=64,
                       hidden_dim=64,
                       geo_feat_dim=15,
                       encoding="hashgrid",
                       bound=float(bound),
                       cuda_ray=False,
                       density_scale=1)
