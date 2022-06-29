import argparse
import os
from autolabel.evaluation import Evaluator
from autolabel.dataset import SceneDataset, LenDataset
from autolabel import utils, model_utils

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scenes', nargs='+')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--vis', action='store_true')
    return parser.parse_args()

def main(flags):
    classes = ["Background", "Class 1"]
    ious = {}
    for scene in flags.scenes:
        scene_name = os.path.basename(os.path.normpath(scene))
        print(f"Evaluating on scene {scene_name}")
        dataset = SceneDataset('test', scene, factor=4.0, batch_size=flags.batch_size)
        model = model_utils.create_model(dataset.min_bounds, dataset.max_bounds).cuda()
        checkpoint_dir = os.path.join(scene, 'nerf', 'checkpoints')
        model_utils.load_checkpoint(model, checkpoint_dir)
        model = model.eval()

        evaluator = Evaluator(model, classes)
        ious[scene_name] = evaluator.eval(dataset, flags.vis)

    from rich.table import Table
    from rich.console import Console
    table = Table()
    table.add_column('Scene')
    for class_name in classes[1:]:
        table.add_column(class_name)

    for scene_name, miou in ious.items():
        row = []
        for class_index, score in miou.items():
            row.append(f"{score:.3f}")
        table.add_row(scene_name, *row)
    console = Console()
    console.print(table)

if __name__ == "__main__":
    main(read_args())

