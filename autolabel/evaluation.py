import cv2
import numpy as np
import os
import cv2
import open3d as o3d
import time
import math
from PIL import Image
import torch
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
            direction_norms = torch.tensor(batch['direction_norms']).to(
                self.device)
            depth = torch.tensor(batch['depth']).to(self.device)
            outputs = self.model.render(rays_o,
                                        rays_d,
                                        direction_norms,
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


def make_legend(axis, semantic_frame, label_mapping):
    classes = np.unique(semantic_frame)
    colors = [COLORS[class_index % COLORS.shape[0]] for class_index in classes]
    prompts = [label_mapping.get(class_id, "unknown") for class_id in classes]
    patches = [
        mpatches.Patch(color=color / 255., label=prompt[:10])
        for color, prompt in zip(colors, prompts)
    ]
    # put those patched as legend-handles into the legend
    axis.legend(handles=patches)


class OpenVocabEvaluator:

    def __init__(self,
                 device='cuda:0',
                 name="model",
                 features=None,
                 checkpoint=None,
                 debug=False,
                 stride=1,
                 save_figures=None,
                 time=False):
        self.device = device
        self.name = name
        self.debug = debug
        self.stride = stride
        self.model = None
        self.label_id_map = None
        self.label_map = None
        self.features = features
        self.extractor = get_feature_extractor(features, checkpoint)
        self.save_figures = save_figures
        self.time = time

    def reset(self, model, label_map, figure_path):
        self.model = model
        self.label_map = label_map
        self.label_id_map = torch.tensor(self.label_map['id'].values).to(
            self.device)
        self.text_features = self._infer_text_features()
        self.label_mapping = {0: 'void'}
        self.label_to_color_id = np.zeros((label_map['id'].max() + 1),
                                          dtype=int)
        for index, (i, prompt) in enumerate(
                zip(label_map['id'], label_map['prompt'])):
            self.label_mapping[i] = prompt
            self.label_to_color_id[i] = index + 1
        self.save_figures = figure_path
        if 'evaluated' in self.label_map:
            self.evaluated_labels = label_map[label_map['evaluated'] ==
                                              1]['id'].values
        else:
            self.evaluated_labels = label_map['id'].values

    def _infer_text_features(self):
        return self.extractor.encode_text(self.label_map['prompt'].values)

    def eval(self, dataset, visualize=False):
        raise NotImplementedError()


class OpenVocabEvaluator2D(OpenVocabEvaluator):

    def eval(self, dataset):
        ious = []
        accs = []
        gt_semantic = dataset.scene.gt_semantic()
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                self.model.eval()
                for i, gt_semantic in enumerate(
                        tqdm(gt_semantic, desc="Evaluating")):
                    if i % self.stride != 0:
                        continue
                    iou = {}
                    acc = {}
                    batch = dataset._get_test(i)
                    gt_semantic = self._read_gt_semantic(
                        gt_semantic, dataset.camera)
                    mask = np.isin(gt_semantic, self.evaluated_labels)

                    p_semantic = self._predict_semantic(batch).cpu().numpy()

                    if self.save_figures is not None:
                        self._save_figure(p_semantic, gt_semantic, batch, i)

                    if self.debug:

                        axis = plt.subplot2grid((1, 2), loc=(0, 0))
                        p_sem = self.label_to_color_id[p_semantic]
                        p_sem_vis = COLORS[p_sem % COLORS.shape[0]]
                        rgb = (batch['pixels'] * 255).astype(np.uint8)
                        axis.imshow(rgb)
                        axis.imshow(p_sem_vis, alpha=0.5)

                        total_iou = float(
                            (p_semantic[mask]
                             == gt_semantic[mask]).sum()) / mask.sum()
                        axis.set_title(f"IoU: {total_iou:.2f}")
                        axis.axis('off')
                        make_legend(axis, p_sem, self.label_mapping)

                        axis = plt.subplot2grid((1, 2), loc=(0, 1))
                        gt = self.label_to_color_id[gt_semantic]
                        axis.imshow(COLORS[gt % COLORS.shape[0]])
                        axis.axis('off')
                        make_legend(axis, gt_semantic, self.label_mapping)
                        plt.tight_layout()
                        plt.show()

                    for i, prompt in zip(self.label_map['id'].values,
                                         self.label_map['prompt'].values):
                        if i not in self.evaluated_labels:
                            continue
                        gt_mask = gt_semantic[mask] == i
                        if gt_mask.sum() <= 0:
                            continue
                        p_mask = p_semantic[mask] == i
                        intersection = np.bitwise_and(p_mask, gt_mask).sum()
                        union = np.bitwise_or(p_mask, gt_mask).sum()
                        if union == 0:
                            class_iou = None
                        else:
                            class_iou = (intersection, union)
                        iou[prompt] = class_iou

                        true_positive = np.bitwise_and(p_mask, gt_mask).sum()
                        true_negative = np.bitwise_and(p_mask == False,
                                                       gt_mask == False).sum()
                        false_positive = np.bitwise_and(p_mask,
                                                        gt_mask == False).sum()
                        false_negative = np.bitwise_and(p_mask == False,
                                                        gt_mask).sum()

                        iou[prompt] = (true_positive, true_positive +
                                       false_positive + false_negative)
                        acc[prompt] = (true_positive,
                                       true_positive + false_positive)
                    ious.append(iou)
                    accs.append(acc)

        if len(ious) == 0:
            print(f"Scene {self.name} has no labels in the evaluation set")
            return {}
        out_iou = {}
        out_acc = {}
        for key in ious[0].keys():
            iou_values = [
                iou[key] for iou in ious if iou.get(key, None) is not None
            ]
            acc_values = [
                acc[key] for acc in accs if acc.get(key, None) is not None
            ]
            if len(iou_values) == 0:
                out_iou[key] = None
                out_acc[key] = None
            else:
                intersection = sum([value[0] for value in iou_values])
                union = sum([value[1] for value in iou_values])
                out_iou[key] = intersection / union
                numerator = sum([value[0] for value in acc_values])
                denominator = sum([value[1] for value in acc_values])
                if denominator == 0:
                    out_acc[key] = 0.0
                else:
                    out_acc[key] = numerator / denominator
        out_iou['total'] = np.mean(list(out_iou.values()))
        out_acc['total'] = np.mean(list(out_acc.values()))
        return out_iou, out_acc

    def _save_figure(self, p_semantic, gt_semantic, batch, example_index):
        rgb_path = os.path.join(self.save_figures, 'rgb')
        p_path = os.path.join(self.save_figures, 'p_semantic')
        gt_path = os.path.join(self.save_figures, 'gt_semantic')
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(p_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)
        rgb = (batch['pixels'] * 255).astype(np.uint8)
        Image.fromarray(rgb).save(
            os.path.join(rgb_path, f"{example_index:06}.png"))
        p_sem = self.label_to_color_id[p_semantic]
        p_sem_vis = COLORS[p_sem % COLORS.shape[0]]
        Image.fromarray(p_sem_vis).save(
            os.path.join(p_path, f"{example_index:06}.png"))
        gt_sem = self.label_to_color_id[gt_semantic]
        gt_sem_vis = COLORS[gt_sem % COLORS.shape[0]]
        gt_sem_vis[gt_semantic == 0] = (0, 0, 0)
        Image.fromarray(gt_sem_vis).save(
            os.path.join(gt_path, f"{example_index:06}.png"))

    def _predict_semantic(self, batch):
        if self.time:
            start = time.time()
        rays_o = torch.tensor(batch['rays_o']).to(self.device)
        rays_d = torch.tensor(batch['rays_d']).to(self.device)
        direction_norms = torch.tensor(batch['direction_norms']).to(self.device)
        depth = torch.tensor(batch['depth']).to(self.device)
        outputs = self.model.render(rays_o,
                                    rays_d,
                                    direction_norms,
                                    staged=True,
                                    perturb=False)
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
        if self.time:
            torch.cuda.synchronize()
            end = time.time()
            n_pixels = H * W
            pixels_per_second = n_pixels / (end - start)
            print(
                f"Semantic prediction for {n_pixels} took {end - start} seconds. {pixels_per_second} pixels per second."
            )
        return self.label_id_map[similarities]

    def _read_gt_semantic(self, path, camera):
        semantic = np.array(
            Image.open(path).resize(camera.size,
                                    Image.NEAREST)).astype(np.int64)
        return semantic


class OpenVocabEvaluator3D(OpenVocabEvaluator):

    def eval(self, dataset, visualize=False):
        point_cloud, gt_semantic = self._read_gt_pointcloud(dataset)
        iou = {}
        acc = {}
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                self.model.eval()
                p_semantic = self._predict_semantic(point_cloud).cpu().numpy()
                mask = np.isin(gt_semantic, self.evaluated_labels)
                intersection = np.bitwise_and(p_semantic == gt_semantic,
                                              mask).sum()

                union = mask.sum()
                if union == 0:
                    print(
                        f"Skipping {self.name} because no labels are in the list of valid labels."
                    )
                    return {}, {}

                if self.debug:
                    pc_vis = point_cloud.cpu().numpy()[mask]
                    pc_vis = o3d.utility.Vector3dVector(pc_vis)
                    pc_vis = o3d.geometry.PointCloud(pc_vis)

                    p_sem = self.label_to_color_id[p_semantic][mask]
                    colors = COLORS[p_sem % COLORS.shape[0]] / 255.
                    pc_vis.colors = o3d.utility.Vector3dVector(colors)

                    o3d.visualization.draw_geometries([pc_vis])

                    gt_sem = self.label_to_color_id[gt_semantic[mask]]
                    gt_colors = COLORS[gt_sem % COLORS.shape[0]] / 255.
                    pc_vis.colors = o3d.utility.Vector3dVector(gt_colors)
                    o3d.visualization.draw_geometries([pc_vis])

                for i, prompt in zip(self.label_map['id'].values,
                                     self.label_map['prompt'].values):
                    if i not in self.evaluated_labels:
                        continue

                    object_mask = gt_semantic[mask] == i
                    if object_mask.sum() == 0:
                        continue
                    p_mask = p_semantic[mask]
                    true_positive = np.bitwise_and(p_mask == i,
                                                   object_mask).sum()
                    true_negative = np.bitwise_and(p_mask != i,
                                                   object_mask == False).sum()
                    false_positive = np.bitwise_and(p_mask == i,
                                                    object_mask == False).sum()
                    false_negative = np.bitwise_and(p_mask != i,
                                                    object_mask).sum()

                    class_iou = float(true_positive) / (
                        true_positive + false_positive + false_negative)
                    iou[prompt] = class_iou
                    acc[prompt] = float(true_positive) / (true_positive +
                                                          false_negative)
        iou['total'] = np.mean(list(iou.values()))
        acc['total'] = np.mean(list(acc.values()))
        return iou, acc

    def _predict_semantic(self, points):
        similarities = torch.zeros(
            (points.shape[0], self.text_features.shape[0]),
            dtype=self.text_features.dtype,
            device=self.device)
        batch_size = 50000
        batches = math.ceil(points.shape[0] / batch_size)
        for batch_index in range(batches):
            batch = points[batch_index * batch_size:(batch_index + 1) *
                           batch_size]
            if self.time:
                start = time.time()
            density = self.model.density(batch)
            _, features = self.model.semantic(density['geo_feat'],
                                              density['sigma'])

            if self.time:
                torch.cuda.synchronize()
                first_batch = time.time()

            N = 10
            scale = 1.0 / N
            for _ in range(N - 1):
                noise = torch.randn_like(batch) * 0.02
                density = self.model.density(batch + noise)
                _, f = self.model.semantic(density['geo_feat'],
                                           density['sigma'])
                features += f * scale
            features = features / torch.norm(features, dim=-1, keepdim=True)
            if self.time:
                torch.cuda.synchronize()
                duration = time.time() - start
                point_count = batch.shape[0] * N
                points_per_sec = point_count / duration
                print(
                    f"Semantic prediction took {duration:.2f} seconds for {point_count} points. {points_per_sec:.2f} points per second."
                )
                duration_ms = (first_batch - start) * 1000
                print(f"Query latency: {duration_ms:.4f} ms")
            for i in range(self.text_features.shape[0]):
                similarities[batch_index * batch_size:(batch_index + 1) *
                             batch_size,
                             i] = (features *
                                   self.text_features[i][None]).sum(dim=-1)
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
        points = torch.tensor(fixed, dtype=torch.float16, device=self.device)
        semantics = semantic.astype(int)

        return points, semantics
