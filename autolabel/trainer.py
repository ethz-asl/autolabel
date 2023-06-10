import os
import tensorboardX
import torch
from torch.nn import functional as F
from torch import optim
import tqdm
from tqdm import tqdm

from torch_ngp.nerf.utils import Trainer

DEPTH_EPSILON = 0.01


class SimpleTrainer(Trainer):

    def train(self, dataloader, epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(
                os.path.join(self.workspace, "run", self.name))

        if self.model.cuda_ray:
            self.model.mark_untrained_grid(dataloader._data.poses,
                                           dataloader._data.intrinsics)

        for i in range(0, epochs):
            self.train_iterations(dataloader, 1000)
            self.epoch += 1

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def train_iterations(self, dataloader, iterations):
        self.model.train()
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(dataloader._data.poses,
                                           dataloader._data.intrinsics)
        iterator = iter(dataloader)
        bar = tqdm(range(iterations), desc="Loss: N/A")
        for _ in bar:
            data = next(iterator)
            for opt in self.optimizers:
                opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                _, _, loss = self.train_step(data)
            self.scaler.scale(loss).backward()
            for opt in self.optimizers:
                self.scaler.step(opt)
            self.scaler.update()
            bar.set_description(f"Loss: {loss:.04f}")
        if self.ema is not None:
            self.ema.update()
        self._step_scheduler(loss)

    def train_step(self, data):
        rays_o = data['rays_o'].to(self.device)  # [B, 3]
        rays_d = data['rays_d'].to(self.device)  # [B, 3]
        direction_norms = data['direction_norms'].to(self.device)  # [B, 1]
        gt_rgb = data['pixels'].to(self.device)  # [B, 3]
        gt_depth = data['depth'].to(self.device)  # [B, 3]
        gt_semantic = data['semantic'].to(self.device)
        has_semantic = gt_semantic >= 0
        use_semantic_loss = has_semantic.sum() > 0

        outputs = self.model.render(rays_o,
                                    rays_d,
                                    direction_norms,
                                    staged=False,
                                    bg_color=None,
                                    perturb=True,
                                    **vars(self.opt))

        pred_rgb = outputs['image']

        loss = self.opt.rgb_weight * self.criterion(pred_rgb, gt_rgb).mean()

        pred_depth = outputs['depth']
        has_depth = (gt_depth > DEPTH_EPSILON)
        depth_loss = torch.abs(pred_depth[has_depth] - gt_depth[has_depth])

        loss = loss + self.opt.depth_weight * depth_loss.mean()

        if self.opt.feature_loss:
            gt_features = data['features'].to(self.device)
            p_features = outputs['semantic_features']
            loss += self.opt.feature_weight * F.l1_loss(
                p_features[:, :gt_features.shape[1]], gt_features)

        pred_semantic = outputs['semantic']
        if use_semantic_loss.item():
            sem_loss = F.cross_entropy(pred_semantic[has_semantic, :],
                                       gt_semantic[has_semantic])
            loss += self.opt.semantic_weight * sem_loss

        return pred_rgb, gt_rgb, loss

    def test_step(self, data):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        direction_norms = data['direction_norms']  # [B, N, 1]
        H, W = data['H'], data['W']

        outputs = self.model.render(rays_o,
                                    rays_d,
                                    direction_norms,
                                    staged=True,
                                    perturb=False,
                                    **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)
        pred_semantic = outputs['semantic']
        pred_features = outputs['semantic_features']
        _, _, C = pred_semantic.shape
        pred_semantic = pred_semantic.reshape(-1, H, W, C)

        return pred_rgb, pred_depth, pred_semantic, pred_features

    def eval_step(self, data):
        rays_o = data['rays_o'].to(self.device)  # [B, 3]
        rays_d = data['rays_d'].to(self.device)  # [B, 3]
        direction_norms = data['direction_norms'].to(self.device)  # [B, 1]
        gt_rgb = data['pixels'].to(self.device)  # [B, H, W, 3]
        gt_depth = data['depth'].to(self.device)  # [B, H, W]
        gt_semantic = data['semantic'].to(self.device)  # [B, H, W]
        H, W, _ = gt_rgb.shape

        outputs = self.model.render(rays_o,
                                    rays_d,
                                    direction_norms,
                                    staged=True,
                                    bg_color=None,
                                    perturb=False,
                                    **vars(self.opt))

        pred_rgb = outputs['image'].reshape(H, W, 3)
        pred_depth = outputs['depth'].reshape(H, W)
        pred_semantic = outputs['semantic']

        loss = self.criterion(pred_rgb, gt_rgb).mean()
        has_depth = gt_depth > DEPTH_EPSILON
        loss += self.opt.depth_weight * torch.abs(pred_depth[has_depth] -
                                                  gt_depth[has_depth]).mean()

        has_semantic = gt_semantic >= 0
        if has_semantic.sum().item() > 0:
            semantic_loss = F.cross_entropy(pred_semantic[has_semantic, :],
                                            gt_semantic[has_semantic])
            loss += self.opt.semantic_weight * semantic_loss

        pred_semantic = pred_semantic.reshape(H, W, pred_semantic.shape[-1])

        return pred_rgb[None], pred_depth[None], pred_semantic[None], gt_rgb[
            None], loss

    def _step_scheduler(self, loss):
        if isinstance(self.lr_schedulers[0],
                      optim.lr_scheduler.ReduceLROnPlateau):
            [s.step(loss) for s in self.lr_schedulers]
        else:
            [s.step() for s in self.lr_schedulers]


class InteractiveTrainer(SimpleTrainer):

    def __init__(self, *args, **kwargs):
        lr_scheduler = kwargs['lr_scheduler']
        kwargs['lr_scheduler'] = None
        super().__init__(*args, **kwargs)
        self.loader = None
        self.lr_scheduler = lr_scheduler(self.optimizer)

    def init(self, loader):
        self.model.train()
        self.iterator = iter(loader)
        self.step = 0
        self.model.mark_untrained_grid(loader._data.poses,
                                       loader._data.intrinsics)

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
