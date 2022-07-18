import argparse
import h5py
import numpy as np
import os
import pickle
import math
from PIL import Image
import torch
from torchvision import transforms
from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms.functional import to_pil_image
from torchvision.models import feature_extraction
from torch.nn import functional as F
from autolabel.utils import Scene
from sklearn import decomposition
from tqdm import tqdm


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--vis', action='store_true')
    return parser.parse_args()


def extract_features(model, scene, output_file):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]).cuda()
    paths = scene.rgb_paths()

    dataset = output_file.create_dataset('fcn_resnet50',
                                         (len(paths), 180, 240, 128),
                                         dtype=np.float16,
                                         compression='lzf')
    features = []
    with torch.inference_mode():
        batch_size = 8
        for i in tqdm(range(math.ceil(len(paths) / batch_size))):
            batch = paths[i * batch_size:(i + 1) * batch_size]
            image = torch.stack([read_image(p) for p in batch]).cuda()
            image = F.interpolate(image, scale_factor=0.5)
            batch = normalize(image / 255.)
            out = model(batch)

            f_small = out['features_small'][:, :64]
            f_large = out['features_large'][:, :64]
            f_small = F.interpolate(f_small,
                                    f_large.shape[-2:],
                                    mode='bilinear')
            features = torch.cat([f_small, f_large],
                                 dim=1).detach().cpu().half().numpy()
            dataset[i * batch_size:(i + 1) * batch_size] = features.transpose(
                [0, 2, 3, 1])

    N, H, W, C = dataset[:].shape
    X = dataset[:].reshape(N * H * W, C)
    pca = decomposition.PCA(n_components=3)
    indices = np.random.randint(0, X.shape[0], size=50000)
    subset = X[indices]
    pca.fit(subset)
    dataset.attrs['pca'] = np.void(pickle.dumps(pca))


def visualize_features(features):
    pca = pickle.loads(features.attrs['pca'].tobytes())
    N, H, W, C = features[:].shape
    X = features[:].reshape(N * H * W, C)
    indices = np.random.randint(0, X.shape[0], size=50000)
    subset = X[indices]
    transformed = pca.transform(subset)
    minimum = transformed.min(axis=0)
    maximum = transformed.max(axis=0)
    diff = maximum - minimum

    from matplotlib import pyplot
    feature_maps = features[:]
    for fm in feature_maps[::10]:
        mapped = pca.transform(fm.reshape(H * W, C)).reshape(H, W, 3)
        normalized = np.clip((mapped - minimum) / diff, 0, 1)
        pyplot.imshow(normalized)
        pyplot.show()


def main():
    flags = read_args()

    scene = Scene(flags.scene)
    output_file = h5py.File(os.path.join(scene.path, 'features.hdf'),
                            'w',
                            libver='latest')
    group = output_file.create_group('features')

    model = fcn_resnet50(pretrained=True)
    model.eval()
    model = model.cuda()
    extractor = feature_extraction.create_feature_extractor(
        model,
        return_nodes={
            'backbone.layer4.2.relu_2': 'features_small',
            'backbone.layer1.2.relu_2': 'features_large'
        })
    extract_features(extractor, scene, group)
    if flags.vis:
        visualize_features(group['fcn_resnet50'])


if __name__ == "__main__":
    main()
