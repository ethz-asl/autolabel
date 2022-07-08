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
from autolabel.dataset import SceneDataset, LenDataset
from autolabel.trainer import SimpleTrainer
from autolabel import model_utils


def read_args():
    parser = model_utils.model_flag_parser()
    parser.add_argument('scene')
    parser.add_argument('--factor-train', type=float, default=2.0)
    parser.add_argument('--factor-test', type=float, default=4.0)
    parser.add_argument('--batch-size', '-b', type=int, default=4096)
    parser.add_argument('--iters', type=int, default=10000)
    parser.add_argument('--workers', '-w', type=int, default=1)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument(
        '--workspace',
        type=str,
        default=None,
        help="Save results in this directory instead of the scene directory.")
    return parser.parse_args()


def write_params(workspace, flags):
    os.makedirs(workspace, exist_ok=True)
    with open(os.path.join(workspace, 'params.pkl'), 'wb') as f:
        pickle.dump(flags, f)


def main():
    flags = read_args()

    dataset = SceneDataset('train',
                           flags.scene,
                           factor=flags.factor_train,
                           batch_size=flags.batch_size)

    model = model_utils.create_model(
        dataset.min_bounds,
        dataset.max_bounds,
        encoding=flags.encoding,
        geometric_features=flags.geometric_features)

    opt = Namespace(rand_pose=-1, color_space='srgb')

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
    model_dir = model_utils.model_dir(flags.scene, flags)
    write_params(model_dir, flags)
    trainer = SimpleTrainer('ngp',
                            opt,
                            model,
                            device='cuda:0',
                            workspace=model_dir,
                            optimizer=optimizer,
                            criterion=criterion,
                            fp16=True,
                            ema_decay=0.95,
                            lr_scheduler=scheduler,
                            scheduler_update_every_step=False,
                            metrics=[],
                            use_checkpoint='latest')
    trainer.train(train_dataloader, epochs)

    testset = SceneDataset('test',
                           flags.scene,
                           factor=flags.factor_test,
                           batch_size=flags.batch_size * 2)
    test_dataloader = torch.utils.data.DataLoader(LenDataset(
        testset, testset.poses.shape[0]),
                                                  batch_size=None,
                                                  num_workers=0)
    trainer.evaluate(test_dataloader)
    trainer.save_checkpoint()


if __name__ == "__main__":
    main()
