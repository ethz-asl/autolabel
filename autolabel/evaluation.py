import torch
import numpy as np
from tqdm import tqdm
from autolabel.constants import COLORS
from rich.progress import track

def compute_iou(p_semantic, gt_semantic, class_index):
    p_semantic = p_semantic == class_index
    gt_semantic = gt_semantic == class_index
    intersection = np.bitwise_and(p_semantic, gt_semantic).sum()
    union = np.bitwise_or(p_semantic, gt_semantic).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)

class Evaluator:
    def __init__(self, model, classes, device='cuda:0'):
        self.model = model
        self.classes = classes
        self.device = device

    def eval(self, dataset, visualize=False):
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                self.model.eval()
                return self._process_frames(dataset, visualize)

    def _process_frames(self, dataset, visualize):
        ious = {}
        gt_masks = dataset.scene.gt_masks(dataset.camera.size)
        for index, gt_semantic in tqdm(gt_masks, desc="Rendering frames"):
            batch = dataset._get_test(index)
            pixels = torch.tensor(batch['pixels']).to(self.device)
            rays_o = torch.tensor(batch['rays_o']).to(self.device)
            rays_d = torch.tensor(batch['rays_d']).to(self.device)
            depth = torch.tensor(batch['depth']).to(self.device)
            outputs = self.model.render(rays_o, rays_d, staged=True, perturb=False)
            p_semantic = outputs['semantic'].argmax(dim=-1).cpu().numpy()
            for class_index in range(1, len(self.classes)):
                if visualize:
                    self._visualize_frame(batch, outputs, gt_semantic)
                iou = compute_iou(p_semantic, gt_semantic, class_index)
                scores = ious.get(class_index, [])
                scores.append(iou)
                ious[class_index] = scores

        for key, scores in ious.items():
            ious[key] = np.mean(scores)
        return ious

    def _visualize_frame(self, batch, outputs, gt_semantic):
        semantic = outputs['semantic'].argmax(dim=-1).cpu().numpy()
        from matplotlib import pyplot
        rgb = (batch['pixels'] * 255).astype(np.uint8)
        axis = pyplot.subplot2grid((1, 2), loc=(0, 0))
        axis.imshow(rgb)
        axis.imshow(COLORS[semantic], alpha=0.5)
        axis = pyplot.subplot2grid((1, 2), loc=(0, 1))
        axis.imshow(COLORS[gt_semantic])
        pyplot.tight_layout()
        pyplot.show()

