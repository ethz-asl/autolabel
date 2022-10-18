import argparse
import os
import numpy as np
import json
from autolabel.evaluation import Evaluator
from autolabel.dataset import SceneDataset, LenDataset
from autolabel import utils, model_utils


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scenes', nargs='+')
    parser.add_argument('--batch-size', default=8182, type=int)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--workspace', type=str, default=None)
    parser.add_argument('--write-images', type=str, default=None)
    parser.add_argument('--out',
                        default=None,
                        type=str,
                        help="Where to write results as json, if anywhere.")
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


def write_results(out, results):
    with open(out, 'wt') as f:
        f.write(json.dumps(results, indent=2))


def main(flags):
    models = gather_models(flags)
    classes = ["Background", "Class 1"]
    scene_names = [os.path.basename(os.path.normpath(p)) for p in flags.scenes]
    scenes = [(s, n) for s, n in zip(flags.scenes, scene_names)]
    scenes = sorted(scenes, key=lambda x: x[1])
    ious = np.ones((len(scenes), len(models))) * -1.
    results = []
    for scene_index, (scene, scene_name) in enumerate(scenes):
        print(f"Evaluating scene {scene_name}")

        nerf_dir = get_nerf_dir(scene, flags)

        for model_hash in models:
            model_path = os.path.join(nerf_dir, model_hash)
            if not os.path.exists(model_path):
                continue
            params = model_utils.read_params(model_path)
            dataset = SceneDataset('test',
                                   scene,
                                   factor=4.0,
                                   batch_size=flags.batch_size,
                                   lazy=True)
            n_classes = dataset.n_classes if dataset.n_classes is not None else 2
            model = model_utils.create_model(dataset.min_bounds,
                                             dataset.max_bounds, n_classes,
                                             params).cuda()
            model = model.eval()

            checkpoint_dir = os.path.join(model_path, 'checkpoints')
            if not os.path.exists(checkpoint_dir) or len(
                    os.listdir(checkpoint_dir)) == 0:
                continue

            model_utils.load_checkpoint(model, checkpoint_dir)
            model = model.eval()

            save_figure_dir = None
            if flags.write_images is not None:
                save_figure_dir = os.path.join(flags.write_images, scene_name)
            evaluator = Evaluator(model,
                                  classes,
                                  name=model_hash,
                                  save_figures=save_figure_dir)
            model_index = models.index(model_hash)
            assert model_index >= 0
            result = evaluator.eval(dataset, flags.vis)

            if len(result.values()) == 0:
                continue
            miou = np.mean([v for v in result.values()])
            assert ious[scene_index, model_index] < 0.0
            ious[scene_index, model_index] = miou
            result = dict(vars(params))
            result['scene'] = scene_name
            result['iou'] = miou
            results.append(result)

    if flags.out is not None:
        write_results(flags.out, results)

    from rich.table import Table
    from rich.console import Console
    table = Table()
    table.add_column('Scene')
    for model in models:
        table.add_column(model)
    for scene_name, scene_ious in zip(scene_names, ious):
        table.add_row(scene_name, *[f"{v:.03f}" for v in scene_ious])
    total_row = ['Total'] + [f"{v:.03f}" for v in ious.mean(axis=0)]
    table.add_row(*total_row, end_section=True)

    console = Console()
    console.print(table)


if __name__ == "__main__":
    main(read_args())
