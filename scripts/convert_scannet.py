description = """
"""
import subprocess
import math
import argparse
import json
import pandas
import zlib
import imageio
from argparse import RawTextHelpFormatter
import os, struct
import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import open3d as o3d

SCANNET20_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
]


def read_args():
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('scannet_scan_dir')
    parser.add_argument(
        '--label-map',
        required=True,
        help="Path to label map .tsv file with semantic label names and ids.")
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-frames',
                        type=int,
                        default=750,
                        help="Maximum number of frames to keep.")
    parser.add_argument('--stride',
                        type=int,
                        default=5,
                        help="Use only every s-th frame.")
    parser.add_argument('--nyu40',
                        action='store_true',
                        help="Use NYU40 label map.")
    parser.add_argument('--scannet20',
                        action='store_true',
                        help="Use ScanNet20 evaluation label map.")
    return parser.parse_args()


class LabelHelper:

    def __init__(self, label_path, flags):
        self.remapping = {}
        self.prompt_remap = {}
        label_map = pandas.read_csv(label_path, sep='\t')
        mapping = np.zeros(label_map['id'].values.max() + 1, np.uint16)
        if flags.nyu40 or flags.scannet20:
            ids = np.arange(1, 41)
            texts = []
            for i in ids:
                text = label_map['nyu40class'][label_map['nyu40id'] ==
                                               i].values[0]
                texts.append(text)
            for i, num in zip(label_map['id'].values,
                              label_map['nyu40id'].values):
                mapping[i] = num
        else:
            texts = label_map['raw_category'].values.tolist()
            ids = np.arange(1, len(texts) + 1)
            for i, num in zip(label_map['id'].values, ids):
                mapping[i] = num

        if flags.scannet20:
            mapping[np.isin(mapping, SCANNET20_IDS) == False] = 0
            texts = [text for text, i in zip(texts, ids) if i in SCANNET20_IDS]
            ids = ids[np.isin(ids, SCANNET20_IDS)]

        self.label_text_to_id = {}
        for num, text in zip(label_map['id'], label_map['raw_category']):
            self.label_text_to_id[text] = num
        self.mapping = mapping

        self.label_map = pandas.DataFrame({'id': ids, 'prompt': texts})
        self.classes_in_scene = set()

    def reset(self):
        self.classes_in_scene = set()

    def _read_config(self, path):
        with open(path, 'rt') as f:
            return json.load(f)

    def write_labelmap(self, out):
        label_map_out = os.path.join(out, 'label_map.csv')
        self.label_map.to_csv(label_map_out, index=False)

    def map_semantics(self, semantic_frame):
        return self.mapping[semantic_frame]

    def register_frame(self, frame):
        for i in np.unique(frame):
            self.classes_in_scene.add(int(i))

    def label_ids(self):
        return self.label_map['id'].values

    def label_to_id(self, label_name):
        scannet_id = self.label_text_to_id[label_name]
        return self.mapping[scannet_id]


def write_intrinsics(out, sensor_reader):
    intrinsics = sensor_reader.intrinsic_color
    intrinsics_path = os.path.join(out, "intrinsics.txt")
    np.savetxt(intrinsics_path, intrinsics)


def write_metadata(out, label_helper):
    metadata_path = os.path.join(out, "metadata.json")
    metadata = {
        "n_classes": int(label_helper.label_ids().max()),
        'classes': list(sorted(label_helper.classes_in_scene))
    }
    with open(metadata_path, 'w') as f:
        f.write(json.dumps(metadata, indent=2))


def read_aggregation(filename):
    """From https://github.com/ScanNet/ScanNet"""
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i][
                'objectId'] + 1  # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    """From https://github.com/ScanNet/ScanNet"""
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def copy_3d_semantics(scene_in, scene, scene_out, label_helper):
    mesh_path = os.path.join(scene_in, f"{scene}_vh_clean_2.ply")
    aggregation = os.path.join(scene_in, f"{scene}.aggregation.json")
    segments = os.path.join(scene_in, f"{scene}_vh_clean_2.0.010000.segs.json")
    mesh = trimesh.load(mesh_path)
    label_ids = np.zeros((mesh.vertices.shape[0],), dtype=np.uint16)
    object_id_to_seg, label_to_segs = read_aggregation(aggregation)
    seg_to_vertex, num_vertices = read_segmentation(segments)
    for label, segs in label_to_segs.items():
        label_id = label_helper.label_to_id(label)
        for seg in segs:
            verts = seg_to_vertex[seg]
            for vertex in verts:
                try:
                    label_ids[vertex] = label_id
                except IndexError:
                    print(
                        f"Index error for {scene} vertex {vertex} and seg: {seg}"
                    )

    out_mesh = os.path.join(scene_out, 'mesh.ply')
    mesh.export(out_mesh)
    out_mesh_semantics = os.path.join(scene_out, 'mesh_labels.npy')
    np.save(out_mesh_semantics, label_ids)


class RGBDFrame():

    def load(self, file_handle):
        self.camera_to_world = np.asarray(struct.unpack(
            'f' * 16, file_handle.read(16 * 4)),
                                          dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = b''.join(
            struct.unpack('c' * self.color_size_bytes,
                          file_handle.read(self.color_size_bytes)))
        self.depth_data = b''.join(
            struct.unpack('c' * self.depth_size_bytes,
                          file_handle.read(self.depth_size_bytes)))


class SensReader:

    def __init__(self, sens_file):
        self.file = sens_file
        self.file_handle = None
        self.num_frames = None
        self.rgb_size = None
        self.depth_size = None

    def __enter__(self):
        self.file_handle = open(self.file, 'rb')
        f = self.file_handle
        version = struct.unpack('I', f.read(4))[0]
        assert version == 4
        strlen = struct.unpack('Q', f.read(8))[0]
        self.sensor_name = ''.join([
            c.decode('utf-8')
            for c in struct.unpack('c' * strlen, f.read(strlen))
        ])
        self.intrinsic_color = np.asarray(struct.unpack('f' * 16,
                                                        f.read(16 * 4)),
                                          dtype=np.float32).reshape(4, 4)
        self.extrinsic_color = np.asarray(struct.unpack('f' * 16,
                                                        f.read(16 * 4)),
                                          dtype=np.float32).reshape(4, 4)
        self.intrinsic_depth = np.asarray(struct.unpack('f' * 16,
                                                        f.read(16 * 4)),
                                          dtype=np.float32).reshape(4, 4)
        self.extrinsic_depth = np.asarray(struct.unpack('f' * 16,
                                                        f.read(16 * 4)),
                                          dtype=np.float32).reshape(4, 4)
        color_compression_type = struct.unpack('i', f.read(4))[0]
        depth_compression_type = struct.unpack('i', f.read(4))[0]
        color_width = struct.unpack('I', f.read(4))[0]
        color_height = struct.unpack('I', f.read(4))[0]
        self.rgb_size = (color_width, color_height)
        depth_width = struct.unpack('I', f.read(4))[0]
        depth_height = struct.unpack('I', f.read(4))[0]
        self.depth_size = (depth_width, depth_height)
        depth_shift = struct.unpack('f', f.read(4))[0]
        self.num_frames = struct.unpack('Q', f.read(8))[0]
        return self

    def __exit__(self, *args):
        self.file_handle.close()

    def read(self):
        for i in range(self.num_frames):
            frame = RGBDFrame()
            frame.load(self.file_handle)
            rgb_frame = imageio.v3.imread(frame.color_data)
            depth_frame = zlib.decompress(frame.depth_data)
            depth_frame = np.frombuffer(depth_frame, dtype=np.uint16).reshape(
                self.depth_size[1], self.depth_size[0])
            yield frame.camera_to_world, rgb_frame, depth_frame


def main():
    flags = read_args()

    os.makedirs(flags.out, exist_ok=True)

    label_helper = LabelHelper(flags.label_map, flags)
    label_helper.write_labelmap(flags.out)

    scenes = os.listdir(flags.scannet_scan_dir)

    for scene in scenes:
        # Reset classes in scene.
        label_helper.reset()
        scene_dir_in = os.path.join(flags.scannet_scan_dir, scene)
        sensor_file = os.path.join(flags.scannet_scan_dir, scene,
                                   f"{scene}.sens")
        semantic_dir_in = os.path.join(flags.scannet_scan_dir, scene,
                                       "label-filt")
        if not os.path.exists(semantic_dir_in):
            label_filt_zip = os.path.join(flags.scannet_scan_dir, scene,
                                          f"{scene}_2d-label-filt.zip")
            subprocess.call(['unzip', label_filt_zip, '-d', scene_dir_in])

        rgb_dir = os.path.join(flags.out, scene, "rgb")
        depth_dir = os.path.join(flags.out, scene, "depth")
        pose_dir = os.path.join(flags.out, scene, "pose")
        semantic_dir = os.path.join(flags.out, scene, "gt_semantic")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(pose_dir, exist_ok=True)
        os.makedirs(semantic_dir, exist_ok=True)

        copy_3d_semantics(os.path.join(flags.scannet_scan_dir, scene), scene,
                          os.path.join(flags.out, scene), label_helper)

        semantic_files = os.listdir(semantic_dir_in)
        semantic_files = sorted(semantic_files,
                                key=lambda x: int(x.split('.')[0]))

        scene_out = os.path.join(flags.out, scene)
        max_frames = 750
        with SensReader(sensor_file) as reader:

            write_intrinsics(scene_out, reader)
            stride = max(math.ceil(reader.num_frames / max_frames),
                         flags.stride)
            for i, ((T_WC, rgb, depth), semantic_file) in enumerate(
                    zip(reader.read(), semantic_files)):
                if i % flags.stride != 0:
                    continue
                print("Processing frame %d" % i, end='\r')
                if np.isnan(T_WC).any() or np.isinf(T_WC).any():
                    print("Skipping frame %d" % i, "because of nan or inf.")
                    continue
                T_CW = np.linalg.inv(T_WC)
                number = f"{i:06}"
                rgb_path = os.path.join(rgb_dir, f"{number}.jpg")
                depth_path = os.path.join(depth_dir, f"{number}.png")
                pose_path = os.path.join(pose_dir, f"{number}.txt")
                imageio.imwrite(rgb_path, rgb)
                cv2.imwrite(depth_path, depth)
                np.savetxt(pose_path, T_CW)

                semantic_path = os.path.join(semantic_dir, f"{number}.png")
                semantic_frame = cv2.imread(
                    os.path.join(semantic_dir_in, semantic_file), -1)
                out_semantic = label_helper.map_semantics(semantic_frame)
                label_helper.register_frame(out_semantic)
                cv2.imwrite(semantic_path, out_semantic)

        write_metadata(scene_out, label_helper)
        subprocess.call([
            'python', 'scripts/compute_scene_bounds.py',
            os.path.join(flags.out, scene)
        ])


if __name__ == "__main__":
    main()
