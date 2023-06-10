"""
This script will export semantic segmentation maps once you are
done with annotating and fitting a scene.

usage: python scripts/export.py <scene1> <scene2> ... --workspace <optional-workspace>

params:
    workspace: The workspace to lookup trained models from.
        Else uses <scene>/nerf/.

Output frames are saved at <scene>/output/semantic/
"""
import cv2
import numpy as np
import os
from skimage import measure
import torch
from tqdm import tqdm

from autolabel import model_utils
from autolabel.dataset import SceneDataset
from autolabel.utils import Scene

MAX_WIDTH = 640


def read_args():
    parser = model_utils.model_flag_parser()
    parser.add_argument('scenes', nargs='+')
    parser.add_argument('--workspace', type=str)
    parser.add_argument('--objects',
                        type=int,
                        default=None,
                        help="""
            If specified, find the specified number of largest connected components per class in the
            produced semantic maps as a post-processing step, removing noise from the segmentation maps.
            """)
    return parser.parse_args()


def lookup_frame_size(scene):
    scene = Scene(scene)
    width, height = scene.peak_image_size()
    if width > MAX_WIDTH:
        scale = MAX_WIDTH / width
        width *= scale
        height *= scale
    return (int(np.round(width)), int(np.round(height)))


def find_largest_components(p_semantic, class_id, object_count):
    p_semantic = p_semantic.copy()
    p_semantic[p_semantic != class_id] = 0
    labels = measure.label(p_semantic)
    counts = np.bincount(labels.flat)[1:]
    largest = []
    sorted_counts = np.argsort(counts)[::-1]
    for i in range(object_count):
        nth_largest_label = sorted_counts[i] + 1
        largest.append(labels == nth_largest_label)
    return largest


def post_process(flags, p_semantic):
    out = np.zeros_like(p_semantic)
    class_ids = np.unique(p_semantic)
    for class_id in class_ids:
        if class_id == 0:
            # Skip background class.
            continue
        components = find_largest_components(p_semantic, class_id,
                                             flags.objects)
        for component in components:
            out[component] = class_id
    return out


def render_frame(model, batch):
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
    return outputs['semantic'].argmax(dim=-1).cpu().numpy()


def export_labels(flags, scene):
    if scene[-1] == os.path.sep:
        scene = scene[:-1]
    scene_name = os.path.basename(scene)
    if flags.workspace is not None:
        model_dir = os.path.join(flags.workspace, scene_name)
    else:
        model_dir = os.path.join(scene, 'nerf')
    models = os.listdir(model_dir)
    if len(models) > 1:
        print(
            f"Warning: scene {scene} has more than 1 model directory. Using {models[0]}."
        )
    elif len(models) == 0:
        print(f"Warning: scene {scene} has no trained models. Skipping.")
        return
    model_dir = os.path.join(model_dir, models[0])
    model_params = model_utils.read_params(model_dir)

    frame_size = lookup_frame_size(scene)

    dataset = SceneDataset('train',
                           scene,
                           size=frame_size,
                           batch_size=16384,
                           features=model_params.features,
                           load_semantic=False)

    n_classes = dataset.n_classes if dataset.n_classes is not None else 2
    model = model_utils.create_model(dataset.min_bounds, dataset.max_bounds,
                                     n_classes, model_params).cuda()
    model = model.eval()
    model_utils.load_checkpoint(model, os.path.join(model_dir, 'checkpoints'))

    output_path = os.path.join(scene, 'output', 'semantic')
    os.makedirs(output_path, exist_ok=True)

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=True):
            for frame_index, rgb_path in zip(tqdm(dataset.indices),
                                             dataset.scene.rgb_paths()):
                batch = dataset._get_test(frame_index)
                frame = render_frame(model, batch)

                if flags.objects is not None:
                    frame = post_process(flags, frame)

                frame_name = os.path.splitext(os.path.basename(rgb_path))[0]
                frame_path = os.path.join(output_path, f"{frame_name}.png")
                cv2.imwrite(frame_path, frame)


def main():
    flags = read_args()

    for scene in flags.scenes:
        export_labels(flags, scene)


if __name__ == "__main__":
    main()
