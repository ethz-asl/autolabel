import torch
import clip
from torch.nn import functional as F
from modules.lseg_module import LSegModule
from additional_utils.models import LSeg_MultiEvalModule
from torchvision import transforms


class LSegFE:

    def __init__(self, checkpoint):
        module = LSegModule.load_from_checkpoint(checkpoint_path=checkpoint,
                                                 backbone='clip_vitl16_384',
                                                 data_path=None,
                                                 num_features=256,
                                                 batch_size=1,
                                                 base_lr=1e-3,
                                                 max_epochs=100,
                                                 augment=False,
                                                 aux=True,
                                                 aux_weight=0,
                                                 ignore_index=255,
                                                 dataset='ade20k',
                                                 se_loss=False,
                                                 se_weight=0,
                                                 arch_option=0,
                                                 block_depth=0,
                                                 activation='lrelu')
        # Skip totensor operation.
        self.transform = transforms.Compose(module.val_transform.transforms[1:])
        net = module.net.cuda()
        scales = [0.75, 1.0, 2.0]
        self.evaluator = LSeg_MultiEvalModule(module, scales=scales,
                                              flip=True).half().cuda().eval()
        self.text_encoder = module.net.clip_pretrained.to(torch.float32).cuda()

    def shape(self, input_shape):
        return (input_shape[0] // 2, input_shape[1] // 2)

    def encode_text(self, text):
        """
        text: list of N strings to encode
        returns: torch tensor size N x 512
        """
        features = self.text_encoder.encode_text(clip.tokenize(text).cuda())
        return features / torch.norm(features, dim=-1, keepdim=True)

    def __call__(self, x):
        x = self.transform(x)
        _, _, H, W = x.shape
        # Return half size features
        H_out, W_out = H // 2, W // 2
        out = []
        for image in x:
            image = image.contiguous()
            out.append(self.evaluator.compute_features(image[None].half()))

        out = torch.cat(out, dim=0)
        out = F.interpolate(out, size=(H_out, W_out), mode='nearest')

        return out.permute(0, 2, 3, 1)
