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
from torch import optim
from autolabel.dataset import SceneDataset, LenDataset
from autolabel.trainer import SimpleTrainer
from autolabel import model_utils


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--factor-train', type=float, default=2.0)
    parser.add_argument('--factor-test', type=float, default=4.0)
    parser.add_argument('--batch-size', '-b', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--iters', type=int, default=10000)
    parser.add_argument('--workers', '-w', type=int, default=1)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--geometric-features', '-g', type=int, default=31)
    parser.add_argument('--encoding',
                        default='hg',
                        choices=['hg', 'hg+freq'],
                        type=str,
                        help="Network positional encoding to use.")
    return parser.parse_args()


def main():
    flags = read_args()

    dataset = SceneDataset('train',
                           flags.scene,
                           factor=flags.factor_train,
                           batch_size=flags.batch_size)

    model = model_utils.create_model(dataset.min_bounds,
                                     dataset.max_bounds,
                                     encoding=flags.encoding,
                                     geometric_features=31)

    opt = Namespace(rand_pose=-1, color_space='srgb')

    optimizer = lambda model: torch.optim.Adam([
        {
            'name': 'encoding',
            'params': list(model.encoder.parameters())
        },
        {
            'name':
                'net',
            'params':
                list(model.sigma_net.parameters()) + list(model.color_net.
                                                          parameters()),
            'weight_decay':
                1e-6
        },
    ],
                                               lr=flags.lr,
                                               betas=(0.9, 0.99),
                                               eps=1e-15)

    train_dataloader = torch.utils.data.DataLoader(LenDataset(dataset, 1000),
                                                   batch_size=None,
                                                   num_workers=flags.workers)
    train_dataloader._data = dataset

    criterion = torch.nn.MSELoss(reduction='none')
    gamma = 0.5
    steps = math.log(1e-4 / flags.lr, gamma)
    step_size = max(flags.iters // steps // 1000, 1)
    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(
        optimizer, gamma=gamma, step_size=step_size)

    epochs = int(np.ceil(flags.iters / 1000))
    workspace = os.path.join(flags.scene, 'nerf', model_utils.model_hash(flags))
    trainer = SimpleTrainer('ngp',
                            opt,
                            model,
                            device='cuda:0',
                            workspace=workspace,
                            optimizer=optimizer,
                            criterion=criterion,
                            fp16=True,
                            ema_decay=0.95,
                            lr_scheduler=scheduler,
                            scheduler_update_every_step=False,
                            metrics=[],
                            use_checkpoint='latest')
    trainer.train(train_dataloader, epochs)
    trainer.evaluate(self, train_dataloader)
    trainer.save_checkpoint()


if __name__ == "__main__":
    main()
