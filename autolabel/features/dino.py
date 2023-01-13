import torch
from torchvision import transforms
from torchvision.io.image import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.models import feature_extraction
from torch.nn import functional as F


class Dino:

    def __init__(self):
        self.normalize = normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).cuda()
        self.model = torch.hub.load('facebookresearch/dino:main',
                                    'dino_vits8').half().cuda()
        self.model.eval()

    def shape(self, *args):
        return (90, 120)

    def __call__(self, x):
        B, C, H, W = x.shape
        x = self.normalize(x)
        x = self.model.get_intermediate_layers(x.half())
        width_out = W // 8
        height_out = H // 8
        return x[0][:, 1:, :].reshape(B, height_out, width_out, 384).detach()
