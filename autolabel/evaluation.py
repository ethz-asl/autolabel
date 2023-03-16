import cv2
import numpy as np
import os
from PIL import Image
import torch
from tqdm import tqdm

from autolabel.constants import COLORS


def compute_iou(p_semantic, gt_semantic, class_index):
    p_semantic = p_semantic == class_index
    gt_semantic = gt_semantic == class_index
    intersection = np.bitwise_and(p_semantic, gt_semantic).sum()
    union = np.bitwise_or(p_semantic, gt_semantic).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


class Evaluator:

    def __init__(self,
                 model,
                 classes,
                 device='cuda:0',
                 name="model",
                 save_figures=None):
        self.model = model
        self.classes = classes
        self.device = device
        self.name = name
        self.save_figures = save_figures

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
            outputs = self.model.render(rays_o,
                                        rays_d,
                                        staged=True,
                                        perturb=False)
            p_semantic = outputs['semantic'].argmax(dim=-1).cpu().numpy()
            for class_index in range(1, len(self.classes)):
                if visualize:
                    self._visualize_frame(batch, outputs, gt_semantic, index)
                iou = compute_iou(p_semantic, gt_semantic, class_index)
                scores = ious.get(class_index, [])
                scores.append(iou)
                ious[class_index] = scores

        for key, scores in ious.items():
            ious[key] = np.mean(scores)
        return ious

    def _visualize_frame(self, batch, outputs, gt_semantic, example_index):
        semantic = outputs['semantic'].argmax(dim=-1).cpu().numpy()
        from matplotlib import pyplot
        rgb = (batch['pixels'] * 255).astype(np.uint8)
        axis = pyplot.subplot2grid((1, 2), loc=(0, 0))
        axis.imshow(rgb)
        p_semantic = COLORS[semantic]
        axis.imshow(p_semantic, alpha=0.5)
        axis.set_title(self.name)
        axis = pyplot.subplot2grid((1, 2), loc=(0, 1))
        axis.imshow(COLORS[gt_semantic])
        axis.set_title("GT")
        pyplot.tight_layout()
        pyplot.show()

        if self.save_figures is not None:
            if not os.path.exists(self.save_figures):
                os.makedirs(self.save_figures, exist_ok=True)
            save_path = os.path.join(self.save_figures,
                                     self.name + f"_{example_index}.jpg")
            image = cv2.addWeighted(rgb, 0.5, p_semantic, 0.5, 0.0)
            Image.fromarray(image).save(save_path)
