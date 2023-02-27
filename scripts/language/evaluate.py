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
    parser.add_argument('--vis-path', type=str, default=None)
    parser.add_argument('--only-scene-classes', action='store_true')
    return parser.parse_args()


def get_nerf_dir(scene, flags):
    scene_name = os.path.basename(os.path.normpath(scene))
    if flags.workspace is None:
        return os.path.join(scene, 'nerf')
    else:
        return os.path.join(flags.workspace, scene_name)


def gather_models(flags, scene_dirs):
    models = set()
    for scene in scene_dirs:
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
    if len(flags.scenes) == 1 and not os.path.exists(
            os.path.join(flags.scenes[0], 'rgb')):
        # We are dealing with a directory full of scenes and not a list of scenes
        scene_dir = flags.scenes[0]
        scene_dirs = [
            os.path.join(scene_dir, scene)
            for scene in os.listdir(scene_dir)
            if os.path.exists(os.path.join(scene_dir, scene, 'rgb'))
        ]
    else:
        scene_dirs = flags.scenes

    original_labels = read_label_map(flags.label_map)
    n_classes = len(original_labels)

    scene_names = [os.path.basename(os.path.normpath(p)) for p in scene_dirs]
    scenes = [(s, n) for s, n in zip(scene_dirs, scene_names)]
    scenes = sorted(scenes, key=lambda x: x[1])
    results = []
    evaluator = None
    for scene_index, (scene, scene_name) in enumerate(scenes):
        model = gather_models(flags, [scene])
        if len(model) == 0:
            print(f"Skipping scene {scene_name} because no models were found.")
            continue
        else:
            model = model[0]
        print(f"Using model {model}")

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
        if flags.only_scene_classes:
            classes_in_scene = dataset.scene.metadata.get('classes', None)
            if classes_in_scene is None:
                label_map = original_labels
            else:
                mask = original_labels['id'].isin(classes_in_scene)
                label_map = original_labels[mask]
        else:
            label_map = original_labels

        model = model_utils.create_model(dataset.min_bounds, dataset.max_bounds,
                                         606, params).cuda()
        model = model.eval()

        checkpoint_dir = os.path.join(model_path, 'checkpoints')
        if not os.path.exists(checkpoint_dir) or len(
                os.listdir(checkpoint_dir)) == 0:
            continue

        model_utils.load_checkpoint(model, checkpoint_dir)
        model = model.eval()
        if flags.vis_path is not None:
            vis_path = os.path.join(flags.vis_path, scene_name)
        else:
            vis_path = None

        if evaluator is None:
            if flags.pc:
                evaluator = OpenVocabEvaluator3D(
                    features=params.features,
                    name=scene_name,
                    checkpoint=flags.feature_checkpoint,
                    stride=flags.stride,
                    debug=flags.debug)
            else:
                evaluator = OpenVocabEvaluator2D(
                    features=params.features,
                    name=scene_name,
                    checkpoint=flags.feature_checkpoint,
                    debug=flags.debug,
                    stride=flags.stride,
                    save_figures=vis_path)
        assert evaluator.features == params.features
        evaluator.reset(model, label_map, vis_path)
        result = evaluator.eval(dataset, flags.vis)

        results.append(result)
        del model

    from rich.table import Table
    from rich.console import Console
    table = Table()
    table.add_column('Class')
    table.add_column('mIoU')

    def iou_to_string(iou):
        if iou is None:
            return "N/A"
        else:
            v = iou * 100
            return f"{v:.1f}"

    reduced = {}
    for result in results:
        for key, value in result.items():
            if key not in reduced:
                reduced[key] = []
            if value is None:
                value = 0.0
            reduced[key].append(value)
    for key, values in reduced.items():
        if key == 'total':
            continue
        mIoU = np.mean(values)
        table.add_row(key, iou_to_string(mIoU))

    scene_total = iou_to_string(
        np.mean([r['total'] for r in results if 'total' in r]))
    table.add_row('Total', scene_total)

    console = Console()
    console.print(table)


if __name__ == "__main__":
    main(read_args())
