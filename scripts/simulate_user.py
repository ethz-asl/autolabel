import os
import math
import argparse
from argparse import Namespace
from os import path
import csv
import queue
import threading
import cv2
import numpy as np
import torch
import pickle
from torch import optim
from tqdm import tqdm
from autolabel.dataset import SceneDataset, LenDataset
from autolabel.trainer import SimpleTrainer
from autolabel.constants import COLORS
from autolabel import model_utils
from matplotlib import pyplot


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
                 clicks_per_step=3,
                 visualize=True,
                 device='cuda:0'):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.clicks_per_step = clicks_per_step
        self.visualize = visualize
        self.semantic_paths = dataset.scene.semantic_paths()
        self.frame_indices = np.arange(0, len(dataset.poses))

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
            chosen_pixel = self._choose_pixel(
                np.bitwise_and(where_defined, where_wrong))
            # if self.visualize:
            #     show_example(p_semantic, gt_semantic, where_wrong, chosen_pixel)
            self._annotate_pixel(frame_index, chosen_pixel, gt_semantic)

    def evaluate(self):
        # Compute how well we are doing
        pass

    def _choose_pixel(self, where_wrong):
        incorrect_indices = np.argwhere(where_wrong)
        return incorrect_indices[np.random.randint(0, len(incorrect_indices))]

    def _infer_semantics(self, frame_index):
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                batch = self.dataset._get_test(frame_index)
                pixels = torch.tensor(batch['pixels']).to(self.device)
                rays_o = torch.tensor(batch['rays_o']).to(self.device)
                rays_d = torch.tensor(batch['rays_d']).to(self.device)
                depth = torch.tensor(batch['depth']).to(self.device)
                outputs = self.model.render(rays_o,
                                            rays_d,
                                            staged=True,
                                            perturb=False)
                return outputs['semantic'].argmax(dim=-1).cpu().numpy()

    def _annotate_pixel(self, frame_index, yx, gt_semantic):
        semantic_class = gt_semantic[yx[0], yx[1]]
        assert semantic_class > 0  # 0 is void class
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
                    depth = torch.tensor(batch['depth']).to(self.device)
                    outputs = self.model.render(rays_o,
                                                rays_d,
                                                staged=True,
                                                perturb=False)
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
                    axis.imshow(rgb)
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
    scheduler = lambda optimizer: optim.lr_scheduler.ConstantLR(optimizer,
                                                                factor=1.0)

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

    user = UserSimulation(model,
                          dataset,
                          visualize=flags.vis,
                          device=trainer.device)

    if flags.vis:
        print("Visualizing at start")
        user.visualize_examples()

    progress = tqdm(range(100), desc='Simulating')
    for i in range(100):
        user.annotate()
        annotated = (dataset.semantics > 0).sum()
        progress.set_description(
            f"Annotation step {i}. {annotated} annotated pixels")
        trainer.train_iterations(train_dataloader, 100)

        if i % 10 == 0 and flags.vis:
            user.visualize_examples()


if __name__ == "__main__":
    main()
