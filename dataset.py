import os
import cv2
import numpy as np
import torch
from PIL import Image
from stray.scene import Scene

CV_TO_OPENGL = np.array([[1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]])

class LazyImageLoader:
    def __init__(self, images, size):
        self.images = images
        self.size = size

    def __getitem__(self, i):
        image = self.images[i]
        frame = np.array(Image.open(image), dtype=np.float32) / 255.
        return cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)

    def __len__(self):
        return len(self.images)

    @property
    def shape(self):
        return [len(self)]

class SceneDataset(torch.utils.data.IterableDataset):
    def __init__(self, split, flags, lazy=False):
        self.lazy = lazy
        self.split = split
        self.batch_size = flags.batch_size
        self.scene = Scene(flags.scene)
        camera = self.scene.camera()
        size = camera.size
        small_size = (int(size[0] / flags.factor), int(size[1] / flags.factor))
        self.indices = np.arange(0, len(self.scene))
        self.resolution = small_size[0] * small_size[1]
        self.camera = self.scene.camera().scale(small_size)
        self.intrinsics = np.array([self.camera.camera_matrix[0, 0], self.camera.camera_matrix[1, 1], self.camera.camera_matrix[0, 2], self.camera.camera_matrix[1, 2]])
        self.scene.camera().camera_matrix
        if split == "train":
            self.next_fn = self._next_train
        else:
            self.next_fn = self._next_test
        self._load_images()
        self._compute_rays()
        self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        self.sample_chunk_size = 32

    def __iter__(self):
        for i in range(len(self)):
            yield self.next_fn(i)

    def __len__(self):
        return self.poses.shape[0]

    def _next_train(self, i):
        chunks = self.batch_size // self.sample_chunk_size
        batch_size = chunks * self.sample_chunk_size
        pixels = np.zeros((batch_size, 3), dtype=np.float32)
        depths = np.zeros(batch_size, dtype=np.float32)
        ray_o = np.zeros((batch_size, 3), dtype=np.float32)
        ray_d = np.zeros((batch_size, 3), dtype=np.float32)

        for chunk in range(chunks):
            start = chunk * self.sample_chunk_size
            end = (chunk+1) * self.sample_chunk_size
            image_index = np.random.randint(0, self.n_examples)
            ray_indices = np.random.randint(0, self.resolution, (self.sample_chunk_size,))

            pixels[start:end] = self.images[image_index][ray_indices]
            depths[start:end] = self.depths[image_index][ray_indices] / 1000.0
            ray_o[start:end] = self.origins[image_index][ray_indices]
            ray_d[start:end] = self.directions[image_index][ray_indices]
        return { 'rays_o': ray_o, 'rays_d': ray_d, 'pixels': pixels, 'depth': depths }

    def _next_test(self, image_index):
        image = self.images[image_index]
        ray_o = self.origins[image_index]
        ray_d = self.directions[image_index]
        depth = self.depths[image_index] / 1000.0
        return { 'pixels': image, 'rays_o': ray_o, 'rays_d': ray_d, 'depth': depth }

    def _load_images(self):
        images = []
        depths = []
        cameras = []

        color_images = self.scene.get_image_filepaths()
        depth_images = self.scene.get_depth_filepaths()

        T_IW = np.load(os.path.join(self.scene.scene_path, 'nerf_transform.npy'))
        poses = [T_IW @ T for T in self.scene.poses]

        for index in self.indices:
            frame = color_images[index]
            if self.lazy:
                images.append(frame)
            else:
                image = np.array(Image.open(frame), dtype=np.float32) / 255.
                image = cv2.resize(image, self.camera.size, interpolation=cv2.INTER_AREA)
                images.append(image)

            depth = cv2.resize(cv2.imread(depth_images[index], -1), self.camera.size, cv2.INTER_NEAREST)
            depths.append(depth)

            T_WC = poses[index] @ CV_TO_OPENGL
            cameras.append(T_WC.astype(np.float32))

        if self.lazy:
            self.images = LazyImageLoader(images, self.camera.size)
        else:
            self.images = np.stack(images, axis=0)
        self.depths = np.stack(depths)
        self.poses = np.stack(cameras, axis=0)
        self.n_examples = self.images.shape[0]
        self.w = self.camera.size[0]
        self.h = self.camera.size[1]

    def _compute_rays(self):
        pixel_center = 0.5
        x, y = np.meshgrid(
                np.arange(self.w, dtype=np.float32) + pixel_center,
                np.arange(self.h, dtype=np.float32) + pixel_center,
                indexing='xy')
        focal_x = self.camera.camera_matrix[0, 0]
        focal_y = self.camera.camera_matrix[1, 1]
        c_x = self.camera.camera_matrix[0, 2]
        c_y = self.camera.camera_matrix[1, 2]
        camera_directions = np.stack([(x - c_x) / focal_x,
            -(y - c_y) / focal_y, -np.ones_like(x)], axis=-1)

        camera_directions = camera_directions / np.linalg.norm(camera_directions, axis=2)[:, :, None]
        directions = ((camera_directions[None, ..., None, :] *
            self.poses[:, None, None, :3, :3]).sum(axis=-1))
        origins = np.broadcast_to(self.poses[:, None, None, :3, -1], directions.shape)
        view_directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

        if self.split == 'train':
            origins = origins.reshape(self.n_examples, self.resolution, 3)
            directions = directions.reshape(self.n_examples, self.resolution, 3)
            view_directions = view_directions.reshape(self.n_examples, self.resolution, 3)
            self.images = self.images.reshape(self.n_examples, self.resolution, 3)
            self.depths = self.depths.reshape(self.n_examples, self.resolution)

        self.origins = origins
        self.directions = directions
        self.viewdirs = view_directions

