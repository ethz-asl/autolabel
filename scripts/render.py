import h5py
import numpy as np
import os
import pickle
from skvideo.io.ffmpeg import FFmpegWriter
import torch
from tqdm import tqdm

from autolabel.constants import COLORS
from autolabel.dataset import SceneDataset
from autolabel import model_utils, visualization


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
    parser.add_argument('--out',
                        type=str,
                        required=True,
                        help="Where to save the video.")
    return parser.parse_args()


class FeatureTransformer:

    def __init__(self, scene_path, features):
        with h5py.File(os.path.join(scene_path, 'features.hdf'), 'r') as f:
            features = f[f'features/{features}']
            blob = features.attrs['pca'].tobytes()
            self.pca = pickle.loads(blob)
            self.feature_min = features.attrs['min']
            self.feature_range = features.attrs['range']

    def __call__(self, p_features):
        H, W, C = p_features.shape
        features = self.pca.transform(p_features.reshape(H * W, C))
        features = np.clip((features - self.feature_min) / self.feature_range,
                           0., 1.)
        return (features.reshape(H, W, 3) * 255.).astype(np.uint8)


def render(model,
           batch,
           dataset,
           feature_transform,
           size=(960, 720),
           maxdepth=10.0):
    rays_o = torch.tensor(batch['rays_o']).cuda()
    rays_d = torch.tensor(batch['rays_d']).cuda()
    direction_norms = torch.tensor(batch['direction_norms']).cuda()
    depth = torch.tensor(batch['depth']).cuda()
    outputs = model.render(rays_o,
                           rays_d,
                           direction_norms,
                           staged=True,
                           perturb=False)
    p_semantic = outputs['semantic'].argmax(dim=-1).cpu().numpy()
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    square_size = (size[0] // 2, size[1] // 2)
    gt_rgb = (batch['pixels'] * 255.0).astype(np.uint8)
    p_depth = outputs['depth']
    frame[:square_size[1], :square_size[0], :] = gt_rgb
    frame[:square_size[1], square_size[0]:] = visualization.visualize_depth(
        p_depth.cpu().numpy(), maxdepth=maxdepth)[:, :, :3]
    frame[square_size[1]:, :square_size[0]] = COLORS[p_semantic]
    p_features = feature_transform(outputs['semantic_features'].cpu().numpy())
    frame[square_size[1]:, square_size[0]:] = p_features
    return frame


def main():
    flags = read_args()

    model_params = model_utils.read_params(flags.model_dir)
    dataset = SceneDataset('train',
                           flags.scene,
                           size=(480, 360),
                           batch_size=16384,
                           features=model_params.features,
                           load_semantic=False)

    n_classes = dataset.n_classes if dataset.n_classes is not None else 2
    model = model_utils.create_model(dataset.min_bounds, dataset.max_bounds,
                                     n_classes, model_params).cuda()
    model = model.eval()
    model_utils.load_checkpoint(model,
                                os.path.join(flags.model_dir, 'checkpoints'))

    feature_transform = FeatureTransformer(flags.scene, model_params.features)
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
                               maxdepth=flags.max_depth)
                writer.writeFrame(frame)
    writer.close()


if __name__ == "__main__":
    main()
