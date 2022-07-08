import os
import torch
import numpy as np
from argparse import Namespace
from PIL import Image
from torch import optim
from autolabel.trainer import SimpleTrainer, InteractiveTrainer
from autolabel.dataset import SceneDataset
from autolabel import model_utils


class TrainingLoop:

    def __init__(self, scene, flags, connection):
        self.flags = flags
        model_hash = model_utils.model_hash(flags)
        self.workspace = os.path.join(scene, 'nerf', model_hash)
        self.train_dataset = SceneDataset('train',
                                          scene,
                                          factor=4.0,
                                          batch_size=flags.batch_size)
        self.model = model_utils.create_model(
            self.train_dataset.min_bounds,
            self.train_dataset.max_bounds,
            encoding=flags.encoding,
            geometric_features=flags.geometric_features)
        self.optimizer = lambda model: torch.optim.Adam([
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
        self.device = 'cuda:0'
        self.fp16 = True
        self._init_trainer()
        self.done = False
        self.connection = connection

    def _init_trainer(self):
        criterion = torch.nn.MSELoss(reduction='none')
        scheduler = lambda optimizer: optim.lr_scheduler.ConstantLR(optimizer,
                                                                    factor=1.0)

        opt = Namespace(rand_pose=-1, color_space='srgb')
        self.trainer = InteractiveTrainer('ngp',
                                          opt,
                                          self.model,
                                          device=self.device,
                                          workspace=self.workspace,
                                          optimizer=self.optimizer,
                                          criterion=criterion,
                                          fp16=self.fp16,
                                          ema_decay=0.95,
                                          lr_scheduler=scheduler,
                                          metrics=[],
                                          use_checkpoint='latest')

    def _create_dataloader(self, dataset):
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=None,
                                             num_workers=0)
        loader._data = self.train_dataset
        return loader

    def run(self):
        self.model.train()
        train_dataloader = self._create_dataloader(self.train_dataset)
        self.trainer.init(train_dataloader)
        while True:
            if self.done:
                break

            self._check_messages()
            self.trainer.take_step()

    def _check_messages(self):
        get_image_message = None
        while self.connection.poll():
            message_type, data = self.connection.recv()
            if message_type == 'update_image':
                self._update_image(data)
            elif message_type == 'get_image':
                get_image_message = data
            elif message_type == 'checkpoint':
                self._save_checkpoint()

        if get_image_message is not None:
            # Only the latest image request is relevant.
            self._get_image(data)

    def _get_image(self, image_index):
        # Image was requested from the other end.
        self.model.eval()
        with torch.no_grad():
            data = self._to_tensor(self.train_dataset._get_test(image_index))
            with torch.cuda.amp.autocast(enabled=self.fp16):
                p_rgb, p_depth, p_semantic = self.trainer.test_step(data)
        self.model.train()
        semantic = p_semantic[0].argmax(dim=-1).detach().cpu()
        self.log(f"Sending {image_index}")
        self.connection.send(("image", {
            'image_index': image_index,
            'rgb': p_rgb[0].detach().cpu(),
            'depth': p_depth[0].detach().cpu(),
            'semantic': semantic
        }))

    def _update_image(self, image_index):
        self.train_dataset.semantic_map_updated(image_index)

    def _save_checkpoint(self):
        checkpoint_path = os.path.join(self.workspace, 'checkpoints')
        name = self.trainer.name
        checkpoint_path = os.path.join(checkpoint_path, f"{name}.pth")
        state = {}
        state['model'] = self.model.state_dict()
        state['optimizer'] = self.trainer.optimizer.state_dict()
        state['scaler'] = self.trainer.scaler.state_dict()
        torch.save(state, checkpoint_path)

    def _to_tensor(self, data):
        dtype = torch.float32
        data['rays_o'] = torch.tensor(data['rays_o'],
                                      device=self.device).to(dtype)
        data['rays_d'] = torch.tensor(data['rays_d'],
                                      device=self.device).to(dtype)
        return data

    def log(self, message):
        print(message)

    def shutdown(self, *args):
        self.done = True
