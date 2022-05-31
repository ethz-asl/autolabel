import json
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
from PIL import Image
from stray.scene import Scene
from torch_ngp.nerf.network_ff import NeRFNetwork
from dataset import SceneDataset
from trainer import SimpleTrainer

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--factor-train', type=float, default=2.0)
    parser.add_argument('--factor-test', type=float, default=4.0)
    parser.add_argument('--batch-size', '-b', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--iters', type=int, default=20000)
    parser.add_argument('--workers', '-w', type=int, default=1)
    return parser.parse_args()

def create_model(dataset):
    extents = dataset.max_bounds - dataset.min_bounds
    bound = (extents - (dataset.min_bounds + dataset.max_bounds) * 0.5).max()
    return NeRFNetwork(num_layers=2, num_layers_color=2,
            hidden_dim_color=64,
            hidden_dim=64,
            geo_feat_dim=15,
            encoding="hashgrid",
            bound=float(bound),
            cuda_ray=False,
            density_scale=1)

class LenDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, length):
        self.dataset = dataset
        self.length = length

    def __iter__(self):
        iterator = iter(self.dataset)
        for _ in range(self.length):
            yield next(iterator)

    def __len__(self):
        return self.length

class TrainingLoop:
    def __init__(self, scene, flags):
        self.flags = flags
        self.workspace = os.path.join(scene, 'nerf')
        self.train_dataset = SceneDataset('train', scene, factor=4.0,
                batch_size=flags.batch_size)
        self.model = create_model(self.train_dataset)
        self.optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': list(model.encoder.parameters())},
            {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
        ], lr=flags.lr, betas=(0.9, 0.99), eps=1e-15)
        self.device = 'cuda:0'
        self.fp16 = True
        self._init_trainer()
        self.done = False

    def _init_trainer(self):
        criterion = torch.nn.MSELoss(reduction='none')
        min_lr = 1e-7
        gamma = 0.9
        scheduler = lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma,
                patience=5,
                min_lr=min_lr)

        opt = Namespace(rand_pose=-1)
        self.trainer = InteractiveTrainer('ngp', opt, self.model,
                device=self.device,
                workspace=self.workspace,
                optimizer=self.optimizer,
                criterion=criterion,
                fp16=self.fp16,
                ema_decay=0.95,
                lr_scheduler=scheduler,
                metrics=[],
                use_checkpoint='latest')

    def run(self, connection):
        self.model.train()
        train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                batch_size=None, num_workers=1)
        train_dataloader._data = self.train_dataset
        self.trainer.init(train_dataloader)
        while True:
            if self.done:
                break
            has_messages = connection.poll()
            if has_messages:
                message = connection.recv()
                image = self._process_message(message)
                connection.send(image)
            loss = self.trainer.take_step()
            print(f"loss: {loss:.04f}", end='\r')

    def _process_message(self, image_index):
        # Image was requested from the other end.
        self.model.eval()
        with torch.no_grad():
            data = self._to_tensor(self.train_dataset._get_test(image_index))
            with torch.cuda.amp.autocast(enabled=self.fp16):
                _, _, p_semantic = self.trainer.test_step(data)
        self.model.train()
        return p_semantic[0].detach().cpu()

    def _to_tensor(self, data):
        dtype = torch.float32
        data['rays_o'] = torch.tensor(data['rays_o'], device=self.device).to(dtype)
        data['rays_d'] = torch.tensor(data['rays_d'], device=self.device).to(dtype)
        return data

    def shutdown(self, *args):
        self.done = True

def main():
    flags = read_args()

    train_dataset = SceneDataset('train', flags.scene, factor=flags.factor_train, batch_size=flags.batch_size)
    val_dataset = SceneDataset('test', flags.scene, factor=flags.factor_test)

    model = create_model(train_dataset)

    opt = Namespace(rand_pose=-1)

    optimizer = lambda model: torch.optim.Adam([
        {'name': 'encoding', 'params': list(model.encoder.parameters())},
        {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
    ], lr=flags.lr, betas=(0.9, 0.99), eps=1e-15)

    train_dataloader = torch.utils.data.DataLoader(LenDataset(train_dataset, 1000),
            batch_size=None, num_workers=flags.workers)
    train_dataloader._data = train_dataset
    val_dataloader = torch.utils.data.DataLoader(LenDataset(val_dataset, len(val_dataset.images)), batch_size=None)
    val_dataloader._data = val_dataset

    criterion = torch.nn.MSELoss(reduction='none')
    min_lr = 1e-7
    gamma = 0.5
    steps = math.log(1e-6, gamma)
    step_size = max(flags.iters // steps // 1000, 1)
    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, gamma=gamma, step_size=step_size)

    epochs = int(np.ceil(flags.iters / 1000))
    workspace = os.path.join(flags.scene, 'nerf')
    trainer = SimpleTrainer('ngp', opt, model,
            device='cuda:0',
            workspace=workspace,
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

if __name__ == "__main__":
    main()

