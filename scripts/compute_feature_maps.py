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
from autolabel.utils.feature_utils import get_feature_extractor
from autolabel.models import Autoencoder
from sklearn import decomposition
from tqdm import tqdm


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--video',
                        type=str,
                        help="Create video of maps and write to this path.")
    parser.add_argument('--features',
                        type=str,
                        choices=['fcn50', 'dino', 'lseg'])
    parser.add_argument('--checkpoint',
                        type=str,
                        help="Which model weights to use.")
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--autoencode', action='store_true')
    return parser.parse_args()


def compress_features(features, dim):
    features = np.stack(features)
    N, H, W, C = features.shape
    coder = Autoencoder(C, dim).cuda()
    optimizer = torch.optim.Adam(coder.parameters(), lr=1e-3)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(features.reshape(N * H * W, C)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=True)
    for _ in range(5):
        bar = tqdm(loader)
        for batch in bar:
            batch = batch[0].cuda()
            reconstructed, code = coder(batch)
            loss = F.mse_loss(reconstructed,
                              batch) + 0.01 * torch.abs(code).mean()
            bar.set_description(f"Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    with torch.inference_mode():
        features_out = np.zeros((N, H, W, dim), dtype=np.float16)
        for i, feature in enumerate(features):
            feature = torch.tensor(feature).view(H * W, C).cuda()
            _, out = coder(feature.view(H * W, C))
            features_out[i] = out.detach().cpu().numpy().reshape(H, W, dim)
    return features_out


def compute_size(image_path, feature):
    image = read_image(image_path)
    _, H, W = image.shape
    short_side = min(H, W)
    if feature in ['fcn50', 'dino']:
        target_size = 720
    elif feature == 'lseg':
        target_size = 242
    scale_factor = target_size / short_side
    return int(H * scale_factor), int(W * scale_factor)


def extract_features(extractor, scene, output_file, flags):
    paths = scene.rgb_paths()
    H, W = compute_size(paths[0], flags.features)

    shape = extractor.shape((H, W))
    dataset = output_file.create_dataset(flags.features,
                                         (len(paths), *shape, flags.dim),
                                         dtype=np.float16,
                                         compression='lzf')

    extracted = []
    with torch.inference_mode():
        batch_size = 2
        for i in tqdm(range(math.ceil(len(paths) / batch_size))):
            index = slice(i * batch_size, (i + 1) * batch_size)
            batch = paths[index]
            image = torch.stack([read_image(p) for p in batch]).cuda()
            image = F.interpolate(image, [H, W])
            features = extractor(image / 255.).cpu().numpy()

            if flags.autoencode:
                extracted += [f for f in features]
            else:
                dataset[index] = features[..., :flags.dim]

    if flags.autoencode:
        features = compress_features(extracted, flags.dim)
        dataset[:] = features

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

    from matplotlib import pyplot
    feature_maps = features[:]
    for fm in feature_maps[::10]:
        mapped = pca.transform(fm.reshape(H * W, C)).reshape(H, W, 3)
        normalized = np.clip(
            (mapped - features.attrs['min']) / features.attrs['range'], 0, 1)
        pyplot.imshow(normalized)
        pyplot.show()


def write_video(features, out):
    from skvideo.io.ffmpeg import FFmpegWriter
    pca = pickle.loads(features.attrs['pca'].tobytes())
    N, H, W, C = features[:].shape
    writer = FFmpegWriter(out,
                          inputdict={'-framerate': '5'},
                          outputdict={
                              '-c:v': 'libx264',
                              '-r': '5',
                              '-pix_fmt': 'yuv420p'
                          })
    for feature in tqdm(features, desc="Encoding frames"):
        mapped = pca.transform(feature.reshape(H * W, C)).reshape(H, W, 3)
        normalized = np.clip(
            (mapped - features.attrs['min']) / features.attrs['range'], 0, 1)
        frame = (normalized * 255.0).astype(np.uint8)
        writer.writeFrame(frame)


def main():
    flags = read_args()
    np.random.seed(0)
    torch.manual_seed(0)

    scene = Scene(flags.scene)
    output_file = h5py.File(os.path.join(scene.path, 'features.hdf'),
                            'w',
                            libver='latest')
    group = output_file.create_group('features')

    extractor = get_feature_extractor(flags.features, flags.checkpoint)

    extract_features(extractor, scene, group, flags)
    if flags.vis:
        visualize_features(group[flags.features])
    if flags.video:
        write_video(group[flags.features], flags.video)


if __name__ == "__main__":
    main()
