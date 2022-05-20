import os
import glob
import tqdm
import math
import random
import warnings

import numpy as np
from torch.nn import functional as F

import time
from datetime import datetime

import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_ngp.nerf.utils import Trainer

DEPTH_EPSILON = 0.01

class SimpleTrainer(Trainer):
    depth_weight = 0.05
    def train_step(self, data):
        rays_o = data['rays_o'].to(self.device) # [B, 3]
        rays_d = data['rays_d'].to(self.device) # [B, 3]
        pixels = data['pixels'].to(self.device) # [B, 3]
        gt_depth = data['depth'].to(self.device) # [B, 3]

        B, C = pixels.shape

        bg_color = None
        gt_rgb = pixels

        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, **vars(self.opt))

        pred_rgb = outputs['image']
        pred_depth = outputs['depth']

        loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]
        # has_depth = (gt_depth > DEPTH_EPSILON).to(pred_rgb.dtype)
        # depth_loss = has_depth * torch.abs(pred_depth - gt_depth)

        loss = loss.mean() # + self.depth_weight * depth_loss.mean()

        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):

        rays_o = data['rays_o'].to(self.device) # [B, 3]
        rays_d = data['rays_d'].to(self.device) # [B, 3]
        pixels = data['pixels'].to(self.device) # [B, H, W, 3]
        gt_depth = data['depth'].to(self.device) # [B, H, W]
        H, W, C = pixels.shape

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=None, perturb=False, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(H, W, 3)
        pred_depth = outputs['depth'].reshape(H, W)

        loss = self.criterion(pred_rgb, pixels).mean()
        # has_depth = (gt_depth > DEPTH_EPSILON).to(pred_rgb.dtype)
        # loss += self.depth_weight * (has_depth * torch.abs(pred_depth - gt_depth)).mean()

        return pred_rgb[None], pred_depth[None], pixels[None], loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth

