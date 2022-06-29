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
    scene = flags.scenes[0]
    dataset = SceneDataset('test', scene, factor=4.0, batch_size=flags.batch_size)
    model = model_utils.create_model(dataset.min_bounds, dataset.max_bounds).cuda()
    checkpoint_dir = os.path.join(scene, 'nerf', 'checkpoints')
    model_utils.load_checkpoint(model, checkpoint_dir)
    model = model.eval()

    classes = ["Background", "Class 1"]
    evaluator = Evaluator(model, classes)
    ious = evaluator.eval(dataset, flags.vis)

    from rich.table import Table
    from rich.console import Console
    table = Table()
    table.add_column('Class')
    table.add_column('mIoU')
    for class_index, miou in ious.items():
        table.add_row(str(class_index), f"{miou:.3f}")
    console = Console()
    console.print(table)

if __name__ == "__main__":
    main(read_args())

