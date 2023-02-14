import cv2
import json
import numpy as np
import os


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

    def write(self, path):
        np.savetxt(path, self.camera_matrix)


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
        image_size = self.peak_image_size()
        if os.path.exists(intrinsics_path):
            self.camera = Camera.from_path(intrinsics_path, image_size)
        self._n_classes = None
        self._metadata = None

    def peak_image_size(self):
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

    def semantic_paths(self):
        return self._get_paths(os.path.join(self.path, 'semantic'))

    def raw_rgb_paths(self):
        return self._get_paths(self.raw_rgb_path)

    def raw_depth_paths(self):
        return self._get_paths(self.raw_depth_path)

    def gt_semantic(self):
        return self._get_paths(os.path.join(self.path, 'gt_semantic'))

    def image_names(self):
        """
        Returns the filenames of rgb images without file extensions.
        """
        rgb_frames = os.listdir(self.rgb_path)
        rgb_frames = sorted(rgb_frames, key=lambda x: int(x.split('.')[0]))
        return [f.split('.')[0] for f in rgb_frames]

    def bbox(self):
        return np.loadtxt(os.path.join(self.path, 'bbox.txt'))[:6].reshape(2, 3)

    def gt_masks(self, size):
        """
        Returns a list of numpy arrays of ground truth segmentation masks,
        if available. Returns an empty list if no masks have been annotated.
        size: the desired size for the masks.
        returns: list of H x W numpy arrays
        """
        gt_masks_dir = os.path.join(self.path, 'gt_masks')
        if not os.path.exists(gt_masks_dir):
            return []
        masks = []
        mask_files = [
            os.path.join(gt_masks_dir, f) for f in os.listdir(gt_masks_dir)
        ]
        for mask_file in mask_files:
            frame_number = int(os.path.basename(mask_file).split('.')[0])
            mask = _read_gt_mask(mask_file, size)
            masks.append((frame_number, _read_gt_mask(mask_file, size)))
        return sorted(masks, key=lambda x: x[0])

    def depth_size(self):
        """
        Return: the size (width, height) of the depth images.
        """
        depth_paths = self.raw_depth_paths()
        if len(depth_paths) == 0:
            depth_paths = self.depth_paths()
        image = cv2.imread(depth_paths[0], -1)
        return (image.shape[1], image.shape[0])

    @property
    def metadata(self):
        if self._metadata is None:
            metadata_path = os.path.join(self.path, 'metadata.json')
            if not os.path.exists(metadata_path):
                return None
            with open(metadata_path) as f:
                self._metadata = json.load(f)
        return self._metadata

    @property
    def n_classes(self):
        if self._n_classes is None:
            self._n_classes = self.metadata['n_classes']
        return self._n_classes


def transform_points(T, points):
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points[..., :, None])[..., :, 0] + t


def _read_gt_mask(path, size):
    image = np.zeros((size[1], size[0]), dtype=np.uint8)
    with open(path, 'rt') as f:
        data = json.load(f)
    scaling_factor = np.array(
        [size[0] / data['imageWidth'], size[1] / data['imageHeight']])
    for shape in data['shapes']:
        polygon = (np.stack(shape['points']) * scaling_factor).astype(np.int32)
        #TODO: handle multiple classes.
        image = cv2.fillPoly(image, polygon[None], 1)
    return image
