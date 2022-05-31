import os
import glob
import tqdm
import math
import random
import time
from tqdm import tqdm
import numpy as np
from datetime import datetime

import cv2

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch_ngp.nerf.utils import Trainer

DEPTH_EPSILON = 0.01

class SimpleTrainer(Trainer):
    depth_weight = 0.05
    semantic_weight = 0.25
    def train_step(self, data):
        rays_o = data['rays_o'].to(self.device) # [B, 3]
        rays_d = data['rays_d'].to(self.device) # [B, 3]
        gt_rgb = data['pixels'].to(self.device) # [B, 3]
        gt_depth = data['depth'].to(self.device) # [B, 3]
        gt_semantic = data['semantic'].to(self.device)
        has_semantic = gt_semantic >= 0
        use_semantic_loss = has_semantic.sum() > 0

        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=None, perturb=True, **vars(self.opt))

        pred_rgb = outputs['image']

        loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]

        pred_depth = outputs['depth']
        has_depth = (gt_depth > DEPTH_EPSILON).to(pred_rgb.dtype)
        depth_loss = has_depth * torch.abs(pred_depth - gt_depth)

        loss = loss.mean() + self.depth_weight * depth_loss.mean()
        pred_semantic = outputs['semantic']
        if use_semantic_loss.item():
            sem_loss = F.cross_entropy(pred_semantic[has_semantic, :], gt_semantic[has_semantic])
            loss += self.semantic_weight * sem_loss

        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):

        rays_o = data['rays_o'].to(self.device) # [B, 3]
        rays_d = data['rays_d'].to(self.device) # [B, 3]
        gt_rgb = data['pixels'].to(self.device) # [B, H, W, 3]
        gt_depth = data['depth'].to(self.device) # [B, H, W]
        gt_semantic = data['semantic'].to(self.device) # [B, H, W]
        H, W, _ = gt_rgb.shape

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=None, perturb=False, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(H, W, 3)
        pred_depth = outputs['depth'].reshape(H, W)
        pred_semantic = outputs['semantic']

        loss = self.criterion(pred_rgb, gt_rgb).mean()
        has_depth = (gt_depth > DEPTH_EPSILON).to(pred_rgb.dtype)
        loss += self.depth_weight * (has_depth * torch.abs(pred_depth - gt_depth)).mean()

        has_semantic = gt_semantic >= 0
        if has_semantic.sum().item() > 0:
            semantic_loss = F.cross_entropy(pred_semantic[has_semantic, :], gt_semantic[has_semantic])
            loss += self.semantic_weight * semantic_loss

        pred_semantic = pred_semantic.reshape(H, W, pred_semantic.shape[-1])

        return pred_rgb[None], pred_depth[None], pred_semantic[None], gt_rgb[None], loss

class InteractiveTrainer(SimpleTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loader = None

    def init(self, loader):
        self.model.train()
        self.iterator = iter(loader)
        self.step = 0
        self.model.mark_untrained_grid(loader._data.poses, loader._data.intrinsics)

    def train(self, loader):
        while True:
            self.model.train()
            self.train_one_epoch(loader)

    def train_one_epoch(self, loader):
        iterator = iter(loader)
        bar = tqdm(range(1000), desc="Loss: N/A")
        for _ in bar:
            data = next(iterator)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                _, _, loss = self.train_step(data)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            bar.set_description(f"Loss: {loss:.04f}")
        if self.ema is not None:
            self.ema.update()
        self._step_scheduler(loss)

    def take_step(self):
        data = next(self.iterator)
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.fp16):
            _, _, loss = self.train_step(data)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.step += 1
        if self.step % 100 == 0:
            self.ema.update()
            self._step_scheduler(loss)
        return loss

    def dataset_updated(self, loader):
        self.loader = loader

    def _step_scheduler(self, loss):
        if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(loss)
        else:
            self.lr_scheduler.step()


