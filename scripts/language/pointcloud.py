import argparse
import os
import numpy as np
import json
import pandas
import torch
from tqdm import tqdm
import open3d as o3d
from autolabel.dataset import SceneDataset, LenDataset
from autolabel import utils, model_utils
from autolabel.utils.feature_utils import get_feature_extractor


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--batch-size', default=8182, type=int)
    parser.add_argument('--workspace', type=str, default=None)
    parser.add_argument('--out',
                        type=str,
                        help="Resulting pointcloud path.",
                        required=True)
    parser.add_argument('--feature-checkpoint', '-f', type=str, required=True)
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help="Only evaluate every Nth frame to save time or for debugging.")
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--features', type=str, default='lseg')
    parser.add_argument('--heatmap',
                        type=str,
                        help="Prompt for generating heatmap.")
    return parser.parse_args()


def get_nerf_dir(scene, flags):
    scene_name = os.path.basename(os.path.normpath(scene))
    if flags.workspace is None:
        return os.path.join(scene, 'nerf')
    else:
        return os.path.join(flags.workspace, scene_name)


def get_model(flags, scene_dir):
    nerf_dir = get_nerf_dir(scene_dir, flags)
    for model in os.listdir(nerf_dir):
        checkpoint_dir = os.path.join(nerf_dir, model, 'checkpoints')
        if os.path.exists(checkpoint_dir):
            return model


def render(model, batch, T_CW, dataset, features):
    rays_o = torch.tensor(batch['rays_o']).cuda()
    rays_d = torch.tensor(batch['rays_d']).cuda()
    direction_norms = torch.tensor(batch['direction_norms']).cuda()
    depth = torch.tensor(batch['depth']).cuda()
    output = model.render(rays_o,
                          rays_d,
                          direction_norms,
                          staged=True,
                          perturb=False,
                          num_steps=512,
                          upsample_steps=0)
    variance = output['depth_variance'].cpu().numpy()
    cutoff = np.percentile(variance, 50)
    mask = variance < cutoff
    cm_C = output['coordinates_map']
    H, W, _ = cm_C.shape
    cm_C = cm_C.cpu().numpy()[mask]
    rgb = output['image'].cpu().numpy()[mask]
    return cm_C[:, :3], rgb


def main(flags):
    scene_name = os.path.basename(os.path.normpath(flags.scene))
    scene = flags.scene
    print(f"Evaluating scene {scene_name}")
    nerf_dir = get_nerf_dir(scene, flags)
    model = get_model(flags, scene)
    model_path = os.path.join(nerf_dir, model)
    params = model_utils.read_params(model_path)
    dataset = SceneDataset('test',
                           scene,
                           factor=4.0,
                           batch_size=flags.batch_size,
                           lazy=True)

    model = model_utils.create_model(dataset.min_bounds, dataset.max_bounds,
                                     606, params).cuda()

    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    if not os.path.exists(checkpoint_dir) or len(
            os.listdir(checkpoint_dir)) == 0:
        print("Now checkpoint path")
        exit()

    model_utils.load_checkpoint(model, checkpoint_dir)
    model = model.eval()
    feature_extractor = get_feature_extractor(flags.features,
                                              flags.feature_checkpoint)

    points = []
    colors = []
    for frame_index in tqdm(dataset.indices[::flags.stride]):
        batch = dataset._get_test(frame_index)
        T_CW = dataset.poses[frame_index]
        points_W, rgb = render(model, batch, T_CW, dataset, feature_extractor)
        points.append(points_W)
        colors.append(rgb)
    rgb = np.concatenate(colors, axis=0)
    p_W = np.concatenate(points, axis=0)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p_W))
    pc.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(flags.out, pc)


if __name__ == "__main__":
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=True):
            main(read_args())
