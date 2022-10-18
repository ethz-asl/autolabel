import torch
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms.functional import to_pil_image
from torchvision.models import feature_extraction
from torch.nn import functional as F


class FCN50:

    def __init__(self):
        self.model = fcn_resnet50(pretrained=True)
        self.model.eval()
        self.model = self.model.cuda()
        self.extractor = feature_extraction.create_feature_extractor(
            self.model, return_nodes={'classifier.2': 'features'})
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]).cuda()

    @property
    def shape(self):
        return (90, 120)

    def __call__(self, x):
        batch = self.normalize(x)
        out = self.extractor(batch)

        return out['features'].detach().cpu().half().numpy().transpose(
            [0, 2, 3, 1])
