import os
import cv2
import numpy as np
import torch
import pickle
from PIL import Image
from stray.scene import Scene
from stray import linalg
from torch_ngp.nerf.provider import nerf_matrix_to_ngp

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
    def __init__(self, split, scene, factor=4.0, batch_size=4096, lazy=False):
        self.lazy = lazy
        self.split = split
        self.batch_size = batch_size
        self.scene = Scene(scene)
        camera = self.scene.camera()
        size = camera.size
        small_size = (int(size[0] / factor), int(size[1] / factor))
        self.indices = np.arange(0, len(self.scene))
        self.resolution = small_size[0] * small_size[1]
        self.camera = self.scene.camera().scale(small_size)
        self.intrinsics = np.array([self.camera.camera_matrix[0, 0], self.camera.camera_matrix[1, 1], self.camera.camera_matrix[0, 2], self.camera.camera_matrix[1, 2]])
        if split == "train":
            self.next_fn = self._next_train
        else:
            self.next_fn = self._next_test
        self._load_images()
        self._compute_rays()
        self.error_map = None
        self.sample_chunk_size = 64

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

        with open(os.path.join(self.scene.scene_path, 'nerf_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        T_IW = metadata['transform']
        poses = [T_IW @ T for T in self.scene.poses]

        for index in self.indices:
            frame = color_images[index]
            if self.lazy:
                images.append(frame)
            else:
                image = np.array(Image.open(frame), dtype=np.float32) / 255.
                image = cv2.resize(image, self.camera.size, interpolation=cv2.INTER_AREA)
                images.append(image)

            T_WC = poses[index] @ CV_TO_OPENGL
            T_WC = nerf_matrix_to_ngp(T_WC, scale=1.0)
            cameras.append(T_WC.astype(np.float32))

            depth_image = cv2.imread(depth_images[index], -1)
            depth = cv2.resize(depth_image, self.camera.size, cv2.INTER_NEAREST)
            depths.append(depth)

        if self.lazy:
            self.images = LazyImageLoader(images, self.camera.size)
        else:
            self.images = np.stack(images, axis=0)

        aabb = metadata['aabb']

        self.depths = np.stack(depths)
        self.poses = np.stack(cameras, axis=0)
        self.n_examples = self.images.shape[0]
        self.w = self.camera.size[0]
        self.h = self.camera.size[1]

        self.min_bounds = aabb[0]
        self.max_bounds = aabb[1]

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
        camera_directions = np.stack([
            (x - c_x) / focal_x,
            (y - c_y) / focal_y,
            np.ones_like(x)], axis=-1)

        camera_directions = camera_directions / np.linalg.norm(camera_directions, axis=-1)[:, :, None]
        camera_directions = camera_directions.reshape(-1, 3) # R x 3
        directions = camera_directions @ self.poses[:, :3, :3].transpose([0, 2, 1])

        origins = np.broadcast_to(self.poses[:, None, :3, -1], directions.shape)

        if self.split == "train":
            self.images = self.images.reshape(self.n_examples, self.resolution, 3)
            self.depths = self.depths.reshape(self.n_examples, self.resolution)
        elif self.split == 'test':
            origins = origins.reshape(self.n_examples, self.h, self.w, 3)
            directions = directions.reshape(self.n_examples, self.h, self.w, 3)

        self.origins = origins
        self.directions = directions

