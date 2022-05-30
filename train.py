import json
import os
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
from PIL import Image
from stray.scene import Scene
from torch_ngp.nerf.network_ff import NeRFNetwork
from dataset import SceneDataset
from trainer import SimpleTrainer

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--factor', type=float, default=4.0)
    parser.add_argument('--batch-size', '-b', type=int, default=4096)
    parser.add_argument('--out', type=str, default='/tmp/train-ngp')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--iters', type=int, default=10000)
    parser.add_argument('--workers', '-w', type=int, default=1)
    return parser.parse_args()

def main():
    flags = read_args()

    train_dataset = SceneDataset('train', flags)
    val_dataset = SceneDataset('test', flags)
    model = NeRFNetwork(bound=2.0, cuda_ray=False, density_scale=10)

    opt = Namespace(rand_pose=-1)

    optimizer = lambda model: torch.optim.Adam([
        {'name': 'encoding', 'params': list(model.encoder.parameters())},
        {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
    ], lr=flags.lr, betas=(0.9, 0.99), eps=1e-15)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None, num_workers=flags.workers, pin_memory=True)
    train_dataloader._data = train_dataset
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=None)
    val_dataloader._data = val_dataset

    criterion = torch.nn.MSELoss(reduction='none')
    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(min(iter / flags.iters, 1), 5))

    epochs = int(np.ceil(flags.iters / len(train_dataloader)))
    trainer = SimpleTrainer('ngp', opt, model,
            device='cuda:0',
            workspace=flags.out,
            optimizer=optimizer,
            criterion=criterion,
            fp16=True,
            ema_decay=0.95,
            lr_scheduler=scheduler,
            scheduler_update_every_step=False,
            metrics=[],
            use_checkpoint='latest',
            eval_interval=10)
    trainer.train(train_dataloader, val_dataloader, epochs)
    trainer.evaluate(val_dataloader)

if __name__ == "__main__":
    main()
