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
        scales = [1.0]
        self.evaluator = LSeg_MultiEvalModule(module, scales=scales,
                                              flip=False).half().cuda().eval()
        self.text_encoder = module.net.clip_pretrained.to(torch.float32).cuda()

    def shape(self, input_shape):
        return (input_shape[0] // 2, input_shape[1] // 2)

    def encode_text(self, text):
        """
        text: list of N strings to encode
        returns: torch tensor size N x 512
        """
        with torch.inference_mode():
            tokenized = clip.tokenize(text).cuda()
            features = []
            for item in tokenized:
                f = self.text_encoder.encode_text(item[None])[0]
                features.append(f)
            features = torch.stack(features, dim=0)
            return features / torch.norm(features, dim=-1, keepdim=True)

    def __call__(self, x):
        x = self.transform(x)
        _, _, H, W = x.shape
        # Return half size features
        H_out, W_out = H // 2, W // 2
        out = []
        x = [F.interpolate(image[None], [H_out, W_out]) for image in x]
        for image in x:
            out.append(self.evaluator.compute_features(image.half()))

        out = torch.cat(out, dim=0)

        return out.permute(0, 2, 3, 1)
