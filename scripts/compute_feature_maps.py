import argparse
import h5py
import numpy as np
import os
import pickle
import math
import torch
from torch.nn import functional as F
from torchvision.io.image import read_image
from PIL import Image
from autolabel.utils import Scene
from sklearn import decomposition
from tqdm import tqdm


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--features', type=str, choices=['fcn50', 'dino'])
    return parser.parse_args()


def extract_features(extractor, scene, output_file, flags):
    paths = scene.rgb_paths()

    dataset = output_file.create_dataset(flags.features,
                                         (len(paths), *extractor.shape, 128),
                                         dtype=np.float16,
                                         compression='lzf')
    with torch.inference_mode():
        batch_size = 8
        for i in tqdm(range(math.ceil(len(paths) / batch_size))):
            batch = paths[i * batch_size:(i + 1) * batch_size]
            image = torch.stack([read_image(p) for p in batch]).cuda()
            image = F.interpolate(image, scale_factor=0.5)
            features = extractor(image / 255.)
            dataset[i * batch_size:(i + 1) * batch_size] = features.transpose(
                [0, 2, 3, 1])

    N, H, W, C = dataset[:].shape
    X = dataset[:].reshape(N * H * W, C)
    pca = decomposition.PCA(n_components=3)
    indices = np.random.randint(0, X.shape[0], size=50000)
    subset = X[indices]
    transformed = pca.fit_transform(subset)
    minimum = transformed.min(axis=0)
    maximum = transformed.max(axis=0)
    diff = maximum - minimum

    dataset.attrs['pca'] = np.void(pickle.dumps(pca))
    dataset.attrs['min'] = minimum
    dataset.attrs['range'] = diff


def visualize_features(features):
    pca = pickle.loads(features.attrs['pca'].tobytes())
    N, H, W, C = features[:].shape
    X = features[:].reshape(N * H * W, C)
    indices = np.random.randint(0, X.shape[0], size=50000)
    subset = X[indices]
    transformed = pca.transform(subset)

    from matplotlib import pyplot
    feature_maps = features[:]
    for fm in feature_maps[::10]:
        mapped = pca.transform(fm.reshape(H * W, C)).reshape(H, W, 3)
        normalized = np.clip((mapped - minimum) / diff, 0, 1)
        pyplot.imshow(normalized)
        pyplot.show()


def get_feature_extractor(features):
    from autolabel.features import FCN50
    if features == 'fcn50':
        return FCN50()
    else:
        raise NotImplementedError()


def main():
    flags = read_args()

    scene = Scene(flags.scene)
    output_file = h5py.File(os.path.join(scene.path, 'features.hdf'),
                            'w',
                            libver='latest')
    group = output_file.create_group('features')

    extractor = get_feature_extractor(flags.features)

    extract_features(extractor, scene, group, flags)
    if flags.vis:
        visualize_features(group['fcn50'])


if __name__ == "__main__":
    main()
