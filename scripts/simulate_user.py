"""
Runs a user-in-the-loop simulation experiment. Requires dense ground truth semantic segmentation maps
to be available.

At each iteration, a frame is selected and inferred using the current model weights. A few incorrectly inferred pixels
are selected and their corresponding ground truth labels are added to the dataset. The model is then fitted for some
iterations before repeating the process.

Between iterations, the accuracy is logged into a csv file scene/nerf/<model-hash>/user_simulation.csv
or <workspace>/<scene_name>/<model-hash>/user_simulation.csv if a workspace is specified.

Usage:
    python scripts/simulate_user.py <scene>
"""
from argparse import Namespace
import cv2
from matplotlib import pyplot
import numpy as np
import os
import torch
from torch import optim

from autolabel import model_utils
from autolabel.constants import COLORS
from autolabel.dataset import SceneDataset
from autolabel.trainer import SimpleTrainer


def read_args():
    parser = model_utils.model_flag_parser()
    parser.add_argument('scene')
    parser.add_argument('--batch-size', '-b', type=int, default=2048)
    parser.add_argument('--workers', '-w', type=int, default=0)
    parser.add_argument('--workspace', type=str, default=None)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--warmup', type=int, default=15000)
    return parser.parse_args()


def show_example(p_semantic, gt_semantic, where_wrong, chosen_pixel):
    axis = pyplot.subplot2grid((1, 3), loc=(0, 0))
    axis.imshow(COLORS[p_semantic])
    axis.scatter(chosen_pixel[1][None], chosen_pixel[0][None], c='red')
    axis = pyplot.subplot2grid((1, 3), loc=(0, 1))
    axis.imshow(COLORS[gt_semantic])
    axis.scatter(chosen_pixel[1][None], chosen_pixel[0][None], c='red')
    axis = pyplot.subplot2grid((1, 3), loc=(0, 2))
    axis.imshow(where_wrong.astype(np.float32))
    axis.scatter(chosen_pixel[1][None], chosen_pixel[0][None], c='red')
    pyplot.tight_layout()
    pyplot.show()


class UserSimulation:

    def __init__(self,
                 model,
                 dataset,
                 result_path,
                 clicks_per_step=5,
                 visualize=True,
                 device='cuda:0'):
        self.model = model
        self.result_path = result_path
        self.device = device
        self.dataset = dataset
        self.clicks_per_step = clicks_per_step
        self.visualize = visualize
        self.semantic_paths = dataset.scene.semantic_paths()
        self.frame_indices = np.arange(0, len(dataset.poses))
        self.evaluation_frames = np.random.choice(self.frame_indices,
                                                  10,
                                                  replace=False)
        self.results = []  # tuple(step, annotated pixels, iou)

    def annotate(self):
        # Sample pixels from those that are incorrect
        # add to the dataset
        # optimize
        frame_index = np.random.choice(self.frame_indices)
        gt_semantic_path = self.semantic_paths[frame_index]
        gt_semantic = self._load_semantic(gt_semantic_path)
        p_semantic = self._infer_semantics(frame_index)
        where_defined = gt_semantic >= 0
        where_wrong = p_semantic != gt_semantic
        for _ in range(self.clicks_per_step):
            chosen_pixel = self._choose_pixel(where_wrong, where_defined)
            self._annotate_pixel(frame_index, chosen_pixel, gt_semantic)
        self.dataset.update_sampler()

    def evaluate(self, current_step, annotated_pixels):
        ious = []
        for index in self.evaluation_frames:
            gt_semantic_path = self.semantic_paths[index]
            gt_semantic = self._load_semantic(gt_semantic_path)
            p_semantic = self._infer_semantics(index)
            where_defined = gt_semantic >= 0
            intersection = gt_semantic == p_semantic
            intersection = np.bitwise_and(where_defined, intersection)
            iou = intersection.sum() / where_defined.sum()
            ious.append(iou)
        miou = np.mean(ious)
        self.results.append((current_step, annotated_pixels, miou))
        return miou

    def save(self):
        np.savetxt(self.result_path, np.array(self.results))

    def _choose_pixel(self, where_wrong, where_defined):
        where_wrong = np.bitwise_and(where_defined, where_wrong)
        if where_wrong.sum() > 0:
            incorrect_indices = np.argwhere(where_wrong)
        else:
            # If all pixels are correct, select one randomly.
            incorrect_indices = np.argwhere(
                np.ones_like(where_wrong, dtype=bool))
        return incorrect_indices[np.random.randint(0, len(incorrect_indices))]

    def _infer_semantics(self, frame_index):
        self.model.eval()
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                batch = self.dataset._get_test(frame_index)
                pixels = torch.tensor(batch['pixels']).to(self.device)
                rays_o = torch.tensor(batch['rays_o']).to(self.device)
                rays_d = torch.tensor(batch['rays_d']).to(self.device)
                direction_norms = torch.tensor(batch['direction_norms']).to(
                    self.device)
                depth = torch.tensor(batch['depth']).to(self.device)
                outputs = self.model.render(rays_o,
                                            rays_d,
                                            direction_norms,
                                            staged=True,
                                            perturb=False)
        self.model.train()
        return outputs['semantic'].argmax(dim=-1).cpu().numpy()

    def _annotate_pixel(self, frame_index, yx, gt_semantic):
        semantic_class = gt_semantic[
            yx[0], yx[1]] + 1  # Counteract shift in _load_semantic
        assert semantic_class >= 0  # -1 is void class
        index = yx[0] * self.dataset.w + yx[1]
        self.dataset.semantics[frame_index][index] = semantic_class

    def _load_semantic(self, path):
        #TODO: check for size and possible post-processing.
        return cv2.imread(path, -1).astype(int) - 1

    def visualize_examples(self):
        n_visualize = 3
        indices_to_visualize = np.random.randint(0, len(self.dataset.poses),
                                                 n_visualize)
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                for i, index in enumerate(indices_to_visualize):
                    batch = self.dataset._get_test(index)
                    pixels = torch.tensor(batch['pixels']).to(self.device)
                    rays_o = torch.tensor(batch['rays_o']).to(self.device)
                    rays_d = torch.tensor(batch['rays_d']).to(self.device)
                    direction_norms = torch.tensor(batch['direction_norms']).to(
                        self.device)
                    depth = torch.tensor(batch['depth']).to(self.device)
                    outputs = self.model.render(rays_o,
                                                rays_d,
                                                direction_norms,
                                                staged=True,
                                                perturb=False)
                    p_rgb = (np.clip(outputs['image'].cpu().numpy(), 0., 1.) *
                             255.).astype(np.uint8)
                    p_semantic = outputs['semantic'].argmax(
                        dim=-1).cpu().numpy()

                    gt_semantic_path = self.semantic_paths[index]
                    gt_semantic = self._load_semantic(gt_semantic_path)
                    gt_semantic[gt_semantic <
                                0] = 0  # Set undefined pixels to background.

                    rgb = (batch['pixels'] * 255).astype(np.uint8)
                    axis = pyplot.subplot2grid((n_visualize, 2), loc=(i, 0))
                    axis.set_title("GT")
                    axis.imshow(rgb)
                    axis.imshow(COLORS[gt_semantic], alpha=0.5)
                    axis.axis('off')
                    axis = pyplot.subplot2grid((n_visualize, 2), loc=(i, 1))
                    axis.set_title("Predicted")
                    axis.imshow(p_rgb)
                    axis.imshow(COLORS[p_semantic], alpha=0.5)
                    axis.axis('off')
        pyplot.tight_layout()
        pyplot.show()


def main():
    flags = read_args()

    dataset = SceneDataset('train',
                           flags.scene,
                           factor=1.0,
                           batch_size=flags.batch_size,
                           features=flags.features,
                           load_semantic=False)

    n_classes = dataset.n_classes if dataset.n_classes is not None else 2
    model = model_utils.create_model(dataset.min_bounds, dataset.max_bounds,
                                     n_classes, flags)
    opt = Namespace(rand_pose=-1,
                    color_space='srgb',
                    feature_loss=flags.features is not None,
                    rgb_weight=flags.rgb_weight,
                    depth_weight=flags.depth_weight,
                    semantic_weight=flags.semantic_weight,
                    feature_weight=flags.feature_weight)
    optimizer = lambda model: torch.optim.Adam([
        {
            'name': 'encoding',
            'params': list(model.encoder.parameters())
        },
        {
            'name': 'net',
            'params': model.network_parameters(),
            'weight_decay': 1e-6
        },
    ],
                                               lr=flags.lr,
                                               betas=(0.9, 0.99),
                                               eps=1e-15)
    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=None,
                                                   num_workers=flags.workers)

    criterion = torch.nn.MSELoss(reduction='none')
    scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(
        optimizer, [10], gamma=0.5)

    model_dir = model_utils.model_dir(flags.scene, flags)
    device = 'cuda:0'
    trainer = SimpleTrainer('ngp',
                            opt,
                            model,
                            device=device,
                            workspace=model_dir,
                            optimizer=optimizer,
                            criterion=criterion,
                            fp16=True,
                            ema_decay=0.95,
                            lr_scheduler=scheduler,
                            scheduler_update_every_step=False,
                            metrics=[],
                            use_checkpoint='latest')
    # Warmup by training without any labels
    trainer.train_iterations(train_dataloader, flags.warmup)

    result_file = os.path.join(model_dir, 'user_simulation.csv')

    np.random.seed(0)

    user = UserSimulation(model,
                          dataset,
                          result_file,
                          visualize=flags.vis,
                          device=trainer.device)

    if flags.vis:
        print("Visualizing at start")
        user.visualize_examples()

    annotated = 0
    i = 0
    while annotated < 1500:
        annotated = (dataset.semantics > 0).sum()
        if i % 5 == 0:
            if flags.vis:
                user.visualize_examples()
            iou = user.evaluate(i, annotated)
            print(f"iou: {iou:.3f}")

        user.annotate()
        print(f"{annotated} annotated pixels")
        trainer.train_iterations(train_dataloader, 250)
        i += 1

    user.save()


if __name__ == "__main__":
    main()
