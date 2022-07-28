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
            self.model,
            return_nodes={
                'backbone.layer4.2.relu_2': 'features_small',
                'backbone.layer1.2.relu_2': 'features_large'
            })
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]).cuda()

    @property
    def shape(self):
        return (180, 240)

    def __call__(self, x):
        batch = self.normalize(x)
        out = self.extractor(batch)

        f_small = out['features_small'][:, :64]
        f_large = out['features_large'][:, :64]
        f_small = F.interpolate(f_small, f_large.shape[-2:], mode='bilinear')
        return torch.cat([f_small, f_large],
                         dim=1).detach().cpu().half().numpy()
