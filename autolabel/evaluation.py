import torch
import numpy as np
from autolabel.constants import COLORS
from rich.progress import track

def compute_iou(mask, gt_mask):
    assert mask.shape == gt_mask.shape
    mask = mask > 0
    gt_mask = gt_mask > 0
    intersection = np.bitwise_and(mask, gt_mask).sum()
    union = np.bitwise_or(mask, gt_mask).sum()
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
        for index in track(dataset.semantic_indices, description="Rendering frames"):
            batch = dataset._get_test(index)
            pixels = torch.tensor(batch['pixels']).to(self.device)
            rays_o = torch.tensor(batch['rays_o']).to(self.device)
            rays_d = torch.tensor(batch['rays_d']).to(self.device)
            depth = torch.tensor(batch['depth']).to(self.device)
            gt_semantic = torch.tensor(batch['semantic']).to(self.device)
            outputs = self.model.render(rays_o, rays_d, staged=True, perturb=False)
            for class_index in range(1, len(self.classes)):
                p_semantic = outputs['semantic'].argmax(dim=-1)
                gt_mask = gt_semantic == class_index
                if gt_mask.sum() == 0:
                    continue
                if visualize:
                    self._visualize_frame(batch, outputs)
                p_mask = p_semantic == class_index
                intersection = torch.bitwise_and(p_mask, gt_mask).sum()
                union = torch.bitwise_or(p_mask, gt_mask).sum()
                iou = float(intersection.item()) / float(union.item())
                scores = ious.get(class_index, [])
                scores.append(iou)
                ious[class_index] = scores

        for key, scores in ious.items():
            ious[key] = np.mean(scores)
        return ious

    def _visualize_frame(self, batch, outputs):
        semantic = outputs['semantic'].argmax(dim=-1).cpu().numpy()
        from matplotlib import pyplot
        gt_rgb = (batch['pixels'] * 255).astype(np.uint8)
        pyplot.imshow(gt_rgb)
        pyplot.imshow(COLORS[semantic], alpha=0.5)
        pyplot.show()

