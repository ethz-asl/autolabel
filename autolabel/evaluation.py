import torch
import numpy as np
import os
import cv2
import open3d as o3d
from PIL import Image
from tqdm import tqdm
from autolabel.constants import COLORS
from autolabel.utils.feature_utils import get_feature_extractor
from autolabel.dataset import CV_TO_OPENGL
from autolabel import utils
from rich.progress import track
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


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


def make_legend(axis, semantic_frame, label_map):
    classes = np.unique(semantic_frame)
    colors = [COLORS[class_index % COLORS.shape[0]] for class_index in classes]
    patches = [
        mpatches.Patch(color=color / 255., label=label_map['prompt'][i][:10])
        for color, i in zip(colors, classes)
    ]
    # put those patched as legend-handles into the legend
    axis.legend(handles=patches)


class OpenVocabEvaluator(Evaluator):

    def __init__(self,
                 device='cuda:0',
                 name="model",
                 features=None,
                 checkpoint=None,
                 debug=False,
                 stride=1):
        self.device = device
        self.name = name
        self.debug = debug
        self.stride = stride
        self.model = None
        self.label_id_map = None
        self.label_map = None
        self.features = features
        self.extractor = get_feature_extractor(features, checkpoint)

    def reset(self, model, label_map):
        self.model = model
        self.label_map = label_map
        self.label_id_map = torch.tensor(self.label_map['id'].values).to(
            self.device)
        self.text_features = self._infer_text_features()

    def _infer_text_features(self):
        return self.extractor.encode_text(self.label_map['prompt'].values)

    def eval(self, dataset, visualize=False):
        raise NotImplementedError()


class OpenVocabEvaluator2D(OpenVocabEvaluator):

    def eval(self, dataset, visualize=False):
        ious = []
        gt_semantic = dataset.scene.gt_semantic()
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                self.model.eval()
                for i, gt_semantic in enumerate(
                        tqdm(gt_semantic, desc="Evaluating")):
                    # For debugging
                    if i % self.stride != 0:
                        continue
                    iou = {}
                    batch = dataset._get_test(i)
                    p_semantic = self._predict_semantic(batch)
                    gt_semantic = self._read_gt_semantic(
                        gt_semantic, dataset.camera)
                    intersection = (p_semantic == gt_semantic).sum().item()
                    union = gt_semantic.numel()
                    iou['total'] = (intersection, union)

                    if self.debug:

                        axis = plt.subplot2grid((1, 2), loc=(0, 0))
                        p_sem = p_semantic.cpu().numpy()
                        p_sem_vis = COLORS[p_sem % COLORS.shape[0]]
                        rgb = (batch['pixels'] * 255).astype(np.uint8)
                        axis.imshow(rgb)
                        axis.imshow(p_sem_vis, alpha=0.5)

                        total_iou = float(intersection) / float(union)
                        axis.set_title(f"IoU: {total_iou:.2f}")
                        axis.axis('off')
                        make_legend(axis, p_sem, self.label_map)

                        axis = plt.subplot2grid((1, 2), loc=(0, 1))

                        gt_sem = gt_semantic.cpu().numpy()
                        axis.imshow(COLORS[gt_sem % COLORS.shape[0]])
                        axis.axis('off')
                        make_legend(axis, gt_sem, self.label_map)
                        plt.tight_layout()
                        plt.show()

                    for i, prompt in zip(self.label_map['id'].values,
                                         self.label_map['prompt'].values):
                        gt_mask = gt_semantic == i
                        p_mask = p_semantic == i
                        intersection = torch.bitwise_and(p_mask,
                                                         gt_mask).sum().item()
                        union = torch.bitwise_or(p_mask, gt_mask).sum().item()
                        if union == 0:
                            class_iou = None
                        else:
                            class_iou = (intersection, union)
                        iou[prompt] = class_iou
                    ious.append(iou)
        out = {}
        for key in ious[0].keys():
            values = [
                iou[key] for iou in ious if iou.get(key, None) is not None
            ]
            if len(values) == 0:
                out[key] = None
            else:
                intersection = sum([value[0] for value in values])
                union = sum([value[1] for value in values])
                out[key] = intersection / union
        return out

    def _predict_semantic(self, batch):
        rays_o = torch.tensor(batch['rays_o']).to(self.device)
        rays_d = torch.tensor(batch['rays_d']).to(self.device)
        depth = torch.tensor(batch['depth']).to(self.device)
        outputs = self.model.render(rays_o, rays_d, staged=True, perturb=False)
        features = outputs['semantic_features']
        features = (features / torch.norm(features, dim=-1, keepdim=True))
        text_features = self.text_features
        H, W, D = features.shape
        C = text_features.shape[0]
        similarities = torch.zeros((H, W, C),
                                   dtype=features.dtype,
                                   device=self.device)
        for i in range(H):
            similarities[i, :, :] = (features[i, :, None] *
                                     text_features).sum(dim=-1)
        similarities = similarities.argmax(dim=-1)
        return self.label_id_map[similarities]

    def _read_gt_semantic(self, path, camera):
        semantic = np.array(
            Image.open(path).resize(camera.size, Image.NEAREST)).astype(
                np.int64) - 1
        return torch.tensor(semantic).to(self.device)


class OpenVocabEvaluator3D(OpenVocabEvaluator):

    def eval(self, dataset, visualize=False):
        point_cloud, gt_semantic = self._read_gt_pointcloud(dataset)
        iou = {}
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                self.model.eval()
                p_semantic = self._predict_semantic(point_cloud)
                intersection = (p_semantic == gt_semantic).sum().item()
                union = gt_semantic.numel()
                iou['total'] = intersection / union

                if self.debug:

                    pc_vis = point_cloud.cpu().numpy()
                    pc_vis = o3d.utility.Vector3dVector(pc_vis)
                    pc_vis = o3d.geometry.PointCloud(pc_vis)

                    p_sem = p_semantic.cpu().numpy()
                    colors = COLORS[p_sem % COLORS.shape[0]] / 255.
                    pc_vis.colors = o3d.utility.Vector3dVector(colors)

                    o3d.visualization.draw_geometries([pc_vis])

                    gt_sem = gt_semantic.cpu().numpy()
                    gt_colors = COLORS[gt_sem % COLORS.shape[0]] / 255.
                    pc_vis.colors = o3d.utility.Vector3dVector(gt_colors)
                    o3d.visualization.draw_geometries([pc_vis])

                for i, prompt in zip(self.label_map['id'].values,
                                     self.label_map['prompt'].values):
                    gt_mask = gt_semantic == i
                    p_mask = p_semantic == i
                    intersection = torch.bitwise_and(p_mask,
                                                     gt_mask).sum().item()
                    union = torch.bitwise_or(p_mask, gt_mask).sum().item()
                    if union == 0:
                        class_iou = None
                    else:
                        class_iou = intersection / union
                    iou[prompt] = class_iou
        return iou

    def _predict_semantic(self, points):
        density = self.model.density(points)
        _, features = self.model.semantic(density['geo_feat'], density['sigma'])
        features = features / torch.norm(features, dim=-1, keepdim=True)
        text_features = self.text_features
        N, D = features.shape
        similarities = torch.zeros((N, text_features.shape[0]),
                                   dtype=text_features.dtype,
                                   device=self.device)
        for i in range(text_features.shape[0]):
            similarities[:, i] = (features * text_features[i][None]).sum(dim=-1)
        similarities = similarities.argmax(dim=-1)
        return self.label_id_map[similarities]

    def _read_gt_pointcloud(self, dataset):
        scene_path = dataset.scene.path
        mesh_path = os.path.join(scene_path, 'mesh.ply')
        gt_semantic_path = os.path.join(scene_path, 'mesh_labels.npy')
        semantic = np.load(gt_semantic_path)
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        points = np.asarray(mesh.vertices)
        fixed = np.zeros_like(points)
        fixed[:, 0] = points[:, 1]
        fixed[:, 1] = points[:, 2]
        fixed[:, 2] = points[:, 0]
        points = torch.tensor(fixed, dtype=torch.float32, device=self.device)
        semantics = torch.tensor(semantic.astype(int),
                                 dtype=torch.long).to(self.device)
        return points, semantics
