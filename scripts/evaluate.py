import argparse
import os
import pickle
import numpy as np
from autolabel.evaluation import Evaluator
from autolabel.dataset import SceneDataset, LenDataset
from autolabel import utils, model_utils


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scenes', nargs='+')
    parser.add_argument('--batch-size', default=8182, type=int)
    parser.add_argument('--vis', action='store_true')
    return parser.parse_args()


def gather_models(scenes):
    models = set()
    for scene in scenes:
        nerf_dir = os.path.join(scene, 'nerf')
        if not os.path.exists(nerf_dir):
            continue
        for model in os.listdir(nerf_dir):
            checkpoint_dir = os.path.join(nerf_dir, model, 'checkpoints')
            if os.path.exists(checkpoint_dir):
                models.add(model)
    return list(models)


def read_params(workspace):
    with open(os.path.join(workspace, 'params.pkl'), 'rb') as f:
        return pickle.load(f)


def main(flags):
    models = gather_models(flags.scenes)
    ious = np.zeros((len(flags.scenes), len(models)))
    classes = ["Background", "Class 1"]
    scene_names = [os.path.basename(os.path.normpath(p)) for p in flags.scenes]
    for scene_index, scene in enumerate(flags.scenes):
        print(f"Evaluating scene {scene}")
        scene_name = scene_names[scene_index]

        nerf_dir = os.path.join(scene, 'nerf')
        models = os.listdir(nerf_dir)

        for model_hash in models:
            model_path = os.path.join(nerf_dir, model_hash)
            params = read_params(model_path)
            dataset = SceneDataset('test',
                                   scene,
                                   factor=4.0,
                                   batch_size=flags.batch_size,
                                   lazy=True)
            model = model_utils.create_model(
                dataset.min_bounds,
                dataset.max_bounds,
                encoding=params.encoding,
                geometric_features=params.geometric_features).cuda()

            checkpoint_dir = os.path.join(model_path, 'checkpoints')
            model_utils.load_checkpoint(model, checkpoint_dir)
            model = model.eval()

            evaluator = Evaluator(model, classes)
            model_index = models.index(model_hash)
            result = evaluator.eval(dataset, flags.vis)
            miou = np.mean([v for v in result.values()])
            ious[scene_index, model_index] = miou

    from rich.table import Table
    from rich.console import Console
    table = Table()
    table.add_column('Scene')
    for model in models:
        table.add_column(model)
    for scene_name, results in zip(scene_names, ious):
        table.add_row(scene_name, *[str(v) for v in results])

    console = Console()
    console.print(table)


if __name__ == "__main__":
    main(read_args())
