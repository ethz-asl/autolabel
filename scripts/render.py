import h5py
import numpy as np
import os
import pickle
from skvideo.io.ffmpeg import FFmpegWriter
import torch
from tqdm import tqdm

from autolabel.constants import COLORS
from autolabel import model_utils
from autolabel import visualization
from autolabel.utils.feature_utils import get_feature_extractor
from matplotlib import pyplot


def read_args():
    parser = model_utils.model_flag_parser()
    parser.add_argument('scene')
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument(
        '--max-depth',
        type=float,
        default=7.5,
        help="The maximum depth used in colormapping the depth frames.")
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--out',
                        type=str,
                        required=True,
                        help="Where to save the video.")
    parser.add_argument('--classes',
                        default=None,
                        type=str,
                        nargs='+',
                        help="Which classes to segment the scene into.")
    parser.add_argument('--label-map',
                        default=None,
                        type=str,
                        help="Path to list of labels.")
    return parser.parse_args()


class FeatureTransformer:

    def __init__(self, scene_path, feature_name, classes, checkpoint=None):
        with h5py.File(os.path.join(scene_path, 'features.hdf'), 'r') as f:
            features = f[f'features/{feature_name}']
            blob = features.attrs['pca'].tobytes()
            self.pca = pickle.loads(blob)
            self.feature_min = features.attrs['min']
            self.feature_range = features.attrs['range']

        if feature_name is not None:
            extractor = get_feature_extractor(feature_name, checkpoint)
            self.text_features = self._encode_text(extractor, classes)

    def _encode_text(self, extractor, text):
        return extractor.encode_text(text)

    def __call__(self, p_features):
        H, W, C = p_features.shape
        features = self.pca.transform(p_features.reshape(H * W, C))
        features = np.clip((features - self.feature_min) / self.feature_range,
                           0., 1.)
        return (features.reshape(H, W, 3) * 255.).astype(np.uint8)


def compute_semantics(outputs, classes, feature_transform):
    if classes is not None:
        features = outputs['semantic_features']
        features = (features / torch.norm(features, dim=-1, keepdim=True))
        text_features = feature_transform.text_features
        H, W, D = features.shape
        C = text_features.shape[0]
        similarities = torch.zeros((H, W, C), dtype=features.dtype)
        for i in range(H):
            similarities[i, :, :] = (features[i, :, None] *
                                     text_features).sum(dim=-1).cpu()
        return similarities.argmax(dim=-1)
    else:
        return outputs['semantic'].argmax(dim=-1).cpu().numpy()


def render(model,
           batch,
           dataset,
           feature_transform,
           size=(960, 720),
           maxdepth=10.0,
           classes=None):
    rays_o = torch.tensor(batch['rays_o']).cuda()
    rays_d = torch.tensor(batch['rays_d']).cuda()
    direction_norms = torch.tensor(batch['direction_norms']).cuda()
    depth = torch.tensor(batch['depth']).cuda()
    outputs = model.render(rays_o,
                           rays_d,
                           direction_norms,
                           staged=True,
                           perturb=False,
                           num_steps=512,
                           upsample_steps=0)
    p_semantic = compute_semantics(outputs, classes, feature_transform)
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    square_size = (size[0] // 2, size[1] // 2)
    p_rgb = (outputs['image'].cpu().numpy() * 255.0).astype(np.uint8)
    p_depth = outputs['depth']
    frame[:square_size[1], :square_size[0], :] = p_rgb
    frame[:square_size[1], square_size[0]:] = visualization.visualize_depth(
        p_depth.cpu().numpy(), maxdepth=maxdepth)[:, :, :3]
    frame[square_size[1]:, :square_size[0]] = COLORS[p_semantic %
                                                     COLORS.shape[0]]
    if feature_transform is not None:
        p_features = feature_transform(
            outputs['semantic_features'].cpu().numpy())
        frame[square_size[1]:, square_size[0]:] = p_features

    return frame


def main():
    flags = read_args()
    model_params = model_utils.read_params(flags.model_dir)

    dataset = SceneDataset('test',
                           flags.scene,
                           size=(480, 360),
                           batch_size=16384,
                           features=model_params.features,
                           load_semantic=False,
                           lazy=True)

    classes = flags.classes
    if flags.label_map is not None:
        label_map = pandas.read_csv(flags.label_map)
        classes_in_scene = dataset.scene.metadata.get('classes', None)
        if classes_in_scene is not None:
            label_map = label_map[label_map['id'].isin(classes_in_scene)]
        classes = label_map['prompt'].values

    feature_transform = None
    if model_params.features is not None:
        feature_transform = FeatureTransformer(flags.scene,
                                               model_params.features, classes,
                                               flags.checkpoint)

    n_classes = dataset.n_classes if dataset.n_classes is not None else 2
    model = model_utils.create_model(dataset.min_bounds, dataset.max_bounds,
                                     n_classes, model_params).cuda()
    model = model.eval()
    model_utils.load_checkpoint(model,
                                os.path.join(flags.model_dir, 'checkpoints'))

    writer = FFmpegWriter(flags.out,
                          inputdict={'-framerate': f'{flags.fps}'},
                          outputdict={
                              '-c:v': 'libx264',
                              '-r': f'{flags.fps}',
                              '-pix_fmt': 'yuv420p'
                          })
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=True):
            for frame_index in tqdm(dataset.indices[::flags.stride]):
                batch = dataset._get_test(frame_index)
                frame = render(model,
                               batch,
                               dataset,
                               feature_transform,
                               maxdepth=flags.max_depth,
                               classes=classes)
                writer.writeFrame(frame)
    writer.close()


if __name__ == "__main__":
    main()
