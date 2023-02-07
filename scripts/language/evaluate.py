import argparse
import os
import numpy as np
import json
import pandas
from autolabel.evaluation import OpenVocabEvaluator2D, OpenVocabEvaluator3D
from autolabel.dataset import SceneDataset, LenDataset
from autolabel import utils, model_utils


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scenes', nargs='+')
    parser.add_argument('--batch-size', default=8182, type=int)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--workspace', type=str, default=None)
    parser.add_argument('--out',
                        default=None,
                        type=str,
                        help="Where to write results as json, if anywhere.")
    parser.add_argument('--label-map', type=str, required=True)
    parser.add_argument('--feature-checkpoint', '-f', type=str, required=True)
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help="Only evaluate every Nth frame to save time or for debugging.")
    parser.add_argument(
        '--pc',
        action='store_true',
        help=
        "Evaluate point cloud segmentation accuracy instead of 2D segmentation maps."
    )
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def get_nerf_dir(scene, flags):
    scene_name = os.path.basename(os.path.normpath(scene))
    if flags.workspace is None:
        return os.path.join(scene, 'nerf')
    else:
        return os.path.join(flags.workspace, scene_name)


def gather_models(flags):
    models = set()
    for scene in flags.scenes:
        nerf_dir = get_nerf_dir(scene, flags)
        if not os.path.exists(nerf_dir):
            continue
        for model in os.listdir(nerf_dir):
            checkpoint_dir = os.path.join(nerf_dir, model, 'checkpoints')
            if os.path.exists(checkpoint_dir):
                models.add(model)
    return list(models)


def read_label_map(path):
    return pandas.read_csv(path)


def write_results(out, results):
    with open(out, 'wt') as f:
        f.write(json.dumps(results, indent=2))


def main(flags):
    model = gather_models(flags)[-1]
    print(f"Using model {model}")

    original_labels = read_label_map(flags.label_map)
    n_classes = len(original_labels)

    scene_names = [os.path.basename(os.path.normpath(p)) for p in flags.scenes]
    scenes = [(s, n) for s, n in zip(flags.scenes, scene_names)]
    scenes = sorted(scenes, key=lambda x: x[1])
    results = []
    for scene_index, (scene, scene_name) in enumerate(scenes):
        print(f"Evaluating scene {scene_name}")

        nerf_dir = get_nerf_dir(scene, flags)
        model_path = os.path.join(nerf_dir, model)
        if not os.path.exists(model_path):
            continue
        params = model_utils.read_params(model_path)
        dataset = SceneDataset('test',
                               scene,
                               factor=4.0,
                               batch_size=flags.batch_size,
                               lazy=True)
        classes_in_scene = dataset.scene.metadata.get('classes', None)
        if classes_in_scene is None:
            label_map = original_labels
        else:
            mask = original_labels['id'].isin(classes_in_scene)
            label_map = original_labels[mask]

        model = model_utils.create_model(dataset.min_bounds, dataset.max_bounds,
                                         dataset.n_classes, params).cuda()
        model = model.eval()

        checkpoint_dir = os.path.join(model_path, 'checkpoints')
        if not os.path.exists(checkpoint_dir) or len(
                os.listdir(checkpoint_dir)) == 0:
            continue

        model_utils.load_checkpoint(model, checkpoint_dir)
        model = model.eval()

        if flags.pc:
            evaluator = OpenVocabEvaluator3D(
                model,
                label_map,
                model_params=params,
                name=scene_name,
                checkpoint=flags.feature_checkpoint,
                stride=flags.stride,
                debug=flags.debug)
        else:
            evaluator = OpenVocabEvaluator2D(
                model,
                label_map,
                model_params=params,
                name=scene_name,
                checkpoint=flags.feature_checkpoint,
                debug=flags.debug,
                stride=flags.stride)
        result = evaluator.eval(dataset, flags.vis)

        results.append(result)

    from rich.table import Table
    from rich.console import Console
    table = Table()
    table.add_column('Scene')

    for scene_name in scene_names:
        table.add_column(scene_name)

    def iou_to_string(iou):
        if iou is None:
            return "N/A"
        else:
            return f"{iou:.2f}"

    for prompt in label_map['prompt'].values:
        scene_results = []
        for i in range(len(scene_names)):
            if prompt not in results[i]:
                scene_results.append("N/A")
                continue
            result = iou_to_string(results[i][prompt])
            scene_results.append(result)
        table.add_row(prompt, *scene_results)

    total_results = []
    for i in range(len(scene_names)):
        scene_total = iou_to_string(results[i]['total'])
        total_results.append(scene_total)
    table.add_row('Mean', *total_results)

    console = Console()
    console.print(table)


if __name__ == "__main__":
    main(read_args())
