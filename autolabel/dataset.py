import os
import cv2
import numpy as np
import torch
import pickle
import random
from PIL import Image
from autolabel.utils import Scene
from torch_ngp.nerf.provider import nerf_matrix_to_ngp

CV_TO_OPENGL = np.array([[1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]])

class LenDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, length):
        self.dataset = dataset
        self.length = length

    def __iter__(self):
        iterator = iter(self.dataset)
        for _ in range(self.length):
            yield next(iterator)

    def __len__(self):
        return self.length


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

class IndexSampler:
    def __init__(self):
        self.classes = np.array([])
        # Index is a dict[int, dict[int, array]]
        # where the first index is the class_id
        # and the second is the image index. The array
        # contains indices which are labeled with that class.
        self.index = {}
        # dict[int, array[float]]
        # where the first int is the class id, the array
        # contains probabilities to sample that class.
        self.image_weights = {}
        self.has_semantics = False
        self.image_range = np.array([])

    def update(self, semantic_maps):
        """
        Recomputes the index.
        0 is the null class, 1 is background and 2 onwards are actual classes.
        """
        assert len(semantic_maps.shape) == 2
        self.index = {}
        self.classes = np.unique(semantic_maps)
        self.classes = self.classes[self.classes != 0] # remove null class
        class_counts = {}
        zeros = np.zeros(len(semantic_maps))
        for i, semantic in enumerate(semantic_maps):
            for class_id in self.classes:
                where_class = semantic == class_id
                if np.any(where_class):
                    self.has_semantics = True
                    image_indices = self.index.get(class_id, {})
                    image_indices[i] = np.argwhere(where_class.ravel()).ravel()
                    self.index[class_id] = image_indices

                    counts = class_counts.get(class_id, zeros).copy()
                    pixel_count = where_class.sum()
                    counts[i] += where_class.sum()
                    class_counts[class_id] = counts

        for class_id, counts in class_counts.items():
            total = counts.sum()
            assert total != 0
            class_counts[class_id] = counts / total
        self.image_weights = class_counts
        self.image_range = np.arange(len(semantic_maps), dtype=int)

    def sample_class(self):
        return np.random.choice(self.classes)

    def sample(self, class_id, count=1):
        """
        Samples an image and {count} pixel indices belonging to class_id in the sampled image.
        The images are sampled proportionally to how many class_id pixels exist in each image.
        returns: tuple(sampled image index, list(sampled pixel index))
        """
        images = self.index[class_id]
        probabilities = self.image_weights[class_id]
        image_index = np.random.choice(self.image_range, p=probabilities)
        pixel_indices = np.random.choice(images[image_index], count)
        return image_index, pixel_indices

    def semantic_indices(self):
        """
        Returns all image indices that have some semantic markings on them.
        """
        indices = set()
        for class_id, semantic_index in self.index.items():
            for index in semantic_index.keys():
                indices.add(index)
        return sorted(list(indices))

class SceneDataset(torch.utils.data.IterableDataset):
    semantic_image_sample_ratio = 0.5
    def __init__(self, split, scene, factor=4.0, batch_size=4096, lazy=False):
        self.lazy = lazy
        self.split = split
        self.batch_size = batch_size
        self.scene = Scene(scene)
        self.index_sampler = IndexSampler()
        camera = self.scene.camera
        size = camera.size
        small_size = (int(size[0] / factor), int(size[1] / factor))
        image_count = len(self.scene)
        self.indices = np.arange(0, image_count)
        self.resolution = small_size[0] * small_size[1]
        self.camera = self.scene.camera.scale(small_size)
        self.intrinsics = np.array([self.camera.camera_matrix[0, 0], self.camera.camera_matrix[1, 1], self.camera.camera_matrix[0, 2], self.camera.camera_matrix[1, 2]])
        self._load_images()
        self._compute_rays()
        self.error_map = None
        self.sample_chunk_size = 32

    def __iter__(self):
        if self.split == "train":
            while True:
                yield self._next_train()
        else:
            for i in range(self.poses.shape[0]):
                yield self._get_test(i)

    def _next_train(self):
        chunks = self.batch_size // self.sample_chunk_size
        batch_size = chunks * self.sample_chunk_size
        pixels = np.zeros((batch_size, 3), dtype=np.float32)
        depths = np.zeros(batch_size, dtype=np.float32)
        semantics = np.zeros(batch_size, dtype=int)
        ray_o = np.zeros((batch_size, 3), dtype=np.float32)
        ray_d = np.zeros((batch_size, 3), dtype=np.float32)

        sampled_indices = np.random.randint(0, self.resolution, (batch_size,))
        for chunk in range(chunks):
            if self.index_sampler.has_semantics and random.random() < self.semantic_image_sample_ratio:
                class_id = self.index_sampler.sample_class()
                image_index, ray_indices = self.index_sampler.sample(class_id, self.sample_chunk_size)
            else:
                image_index = np.random.randint(0, self.n_examples)
                ray_indices = np.random.randint(0, self.resolution, (self.sample_chunk_size,))
            start = chunk * self.sample_chunk_size
            end = (chunk+1) * self.sample_chunk_size

            pixels[start:end] = self.images[image_index][ray_indices]
            depths[start:end] = self.depths[image_index][ray_indices] / 1000.0
            semantics[start:end] = self.semantics[image_index][ray_indices].astype(int) - 1
            ray_o[start:end] = np.broadcast_to(self.origins[image_index, None], (ray_indices.shape[0], 3))
            ray_d[start:end] = self.directions[image_index][ray_indices]
        return { 'rays_o': ray_o, 'rays_d': ray_d, 'pixels': pixels, 'depth': depths, 'semantic': semantics }

    def _get_test(self, image_index):
        image = self.images[image_index].reshape(self.h, self.w, 3)
        ray_o = np.broadcast_to(self.origins[image_index], (self.h, self.w, 3))
        ray_d = self.directions[image_index].reshape(self.h, self.w, 3)
        depth = (self.depths[image_index] / 1000.0).reshape(self.h, self.w)
        semantic = (self.semantics[image_index].astype(int) - 1).reshape(self.h, self.w)
        return { 'pixels': image, 'rays_o': ray_o,
                'rays_d': ray_d, 'depth': depth, 'semantic': semantic,
                'H': self.h, 'W': self.w }

    def _load_images(self):
        images = []
        depths = []
        semantics = []
        cameras = []

        color_images = self.scene.rgb_paths()
        depth_images = self.scene.depth_paths()

        poses = self.scene.poses

        for index in self.indices:
            frame = color_images[index]
            if self.lazy:
                images.append(frame)
            else:
                image = np.array(Image.open(frame), dtype=np.float32) / 255.
                image = cv2.resize(image, self.camera.size, interpolation=cv2.INTER_AREA)
                images.append(image)

            semantic_path = os.path.join(self.scene.path, 'semantic', os.path.basename(depth_images[index]))
            if os.path.exists(semantic_path):
                image = Image.open(semantic_path)
                image = image.resize(self.camera.size, Image.NEAREST)
                semantics.append(np.asarray(image))
            else:
                semantics.append(np.zeros(self.camera.size[::-1], dtype=np.uint8))

            T_CW = poses[index]
            T_WC = np.linalg.inv(T_CW) @ CV_TO_OPENGL
            T_WC = nerf_matrix_to_ngp(T_WC, scale=1.0)
            cameras.append(T_WC.astype(np.float32))

            depth_image = cv2.imread(depth_images[index], -1)
            depth = cv2.resize(depth_image, self.camera.size, cv2.INTER_NEAREST)
            depths.append(depth)

        if self.lazy:
            self.images = LazyImageLoader(images, self.camera.size)
        else:
            self.images = np.stack(images, axis=0)

        aabb = self.scene.bbox()

        self.depths = np.stack(depths)
        self.semantics = np.stack(semantics)
        self.index_sampler.update(self.semantics.reshape(-1, self.resolution))
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
        focal_x = self.camera.fx
        focal_y = self.camera.fy
        c_x = self.camera.cx
        c_y = self.camera.cy
        camera_directions = np.stack([
            (x - c_x) / focal_x,
            (y - c_y) / focal_y,
            np.ones_like(x)], axis=-1)

        camera_directions = camera_directions / np.linalg.norm(camera_directions, axis=-1)[:, :, None]
        camera_directions = camera_directions.reshape(-1, 3) # R x 3
        directions = (self.poses[:, None, :3, :3] @ camera_directions[None, :, :, None])[:, :, :, 0]

        self.origins = self.poses[:, :3, -1]

        if self.split == "train":
            self.images = self.images.reshape(self.n_examples, self.resolution, 3)
            self.depths = self.depths.reshape(self.n_examples, self.resolution)
            self.semantics = self.semantics.reshape(self.n_examples, self.resolution)
        elif self.split == 'test':
            directions = directions.reshape(self.n_examples, self.h, self.w, 3)

        self.directions = directions

    def semantic_map_updated(self, image_index):
        semantic_path = os.path.join(self.scene.path, 'semantic', f"{image_index}.png")
        if os.path.exists(semantic_path):
            image = Image.open(semantic_path)
            image = np.asarray(image.resize(self.camera.size, Image.NEAREST))
            self.semantics[image_index, :] = image.reshape(self.resolution)
            self.index_sampler.update(self.semantics)
        else:
            print(f"Could not find image {semantic_path}")


