import torch
import glob
import argparse
import os
from autolabel.models import ALNetwork


def load_checkpoint(model, checkpoint_dir, device='cuda:0'):
    checkpoint_list = sorted(glob.glob(f'{checkpoint_dir}/*.pth'))
    checkpoint = checkpoint_list[-1]
    checkpoint_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_dict['model'])
    return model


def model_flag_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--geometric-features', '-g', type=int, default=15)
    parser.add_argument('--encoding',
                        default='hg',
                        choices=['hg', 'hg+freq'],
                        type=str,
                        help="Network positional encoding to use.")
    return parser


def model_hash(flags):
    return f"g{flags.geometric_features}_{flags.encoding}"


def model_dir(scene_path, flags):
    mhash = model_hash(flags)
    if flags.workspace is None:
        return os.path.join(scene_path, 'nerf', mhash)
    scene_name = os.path.basename(os.path.normpath(flags.scene))
    return os.path.join(flags.workspace, scene_name, mhash)


def create_model(min_bounds, max_bounds, encoding='hg', geometric_features=31):
    extents = max_bounds - min_bounds
    bound = (extents - (min_bounds + max_bounds) * 0.5).max()
    return ALNetwork(num_layers=2,
                     num_layers_color=2,
                     hidden_dim_color=64,
                     hidden_dim=64,
                     geo_feat_dim=geometric_features,
                     encoding=encoding,
                     bound=float(bound),
                     cuda_ray=False,
                     density_scale=1)
