description = """
This script converts scenes from the ARKitScenes dataset (https://github.com/apple/ARKitScenes) format to
the format used by autolabel.

Usage:
    python scripts/convert_arkitscenes.py <path-to-arkit-scenes-dir> --out <path-to-output-dir>

After running this script, scripts/compute_scene_bounds.py needs to be run to compute the scene bounding box.

This script uses the lowres_wide, lowres_depth, lowres_wide.traj, confidence, lowres_wide_intrinsics parts of the dataset.

See Apple's instructions here for details https://github.com/apple/ARKitScenes/blob/main/DATA.md.

The script to download the ARKitScenes dataset can be found here https://github.com/apple/ARKitScenes/blob/main/download_data.py.

To download the required parts use it like this:
python download_data.py raw --split Training --video_id_csv depth_upsampling/upsampling_train_val_splits.csv --download_dir /tmp/arkit_scenes/ --raw_dataset_assets lowres_wide lowres_depth lowres_wide.traj confidence lowres_wide_intrinsics

"""
import argparse
from argparse import RawTextHelpFormatter
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def read_args():
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('arkit_scenes')
    parser.add_argument('--out')
    return parser.parse_args()


def read_trajectory(path):
    return np.loadtxt(path)


def extract_name(filename):
    return filename.replace('.png', '')


def collect_images(dir_path):
    filenames = os.listdir(dir_path)
    out = {}
    for filename in filenames:
        name = extract_name(filename)
        out[name] = os.path.join(dir_path, filename)
    return out


def read_intrinsics(dir_path):
    intrinsic_files = os.listdir(dir_path)
    intrinsic_path = os.path.join(dir_path, intrinsic_files[0])
    _, _, fx, fy, cx, cy = np.loadtxt(intrinsic_path)
    C = np.eye(3)
    C[0, 0] = fx
    C[1, 1] = fy
    C[0, 2] = cx
    C[1, 2] = cy
    return C


def to_ts(filename):
    _, ts = filename.split('_')
    seconds, ms = [int(v) for v in ts.split('.')]
    return seconds + ms * 1e-3


def find_pose(trajectory, rgb_name):
    timestamp = to_ts(rgb_name)
    errors = np.abs(trajectory[:, 0] - timestamp)
    closest = errors.argmin()
    return trajectory[closest], errors[closest]


def to_transform(pose):
    rotvec = pose[1:4]
    translation = pose[4:]
    T_CW = np.eye(4)
    R_CW = Rotation.from_rotvec(rotvec)
    T_CW[:3, :3] = R_CW.as_matrix()
    T_CW[:3, 3] = translation
    return T_CW


def write_scene(flags, scene_name, trajectory, rgb_images, depth_images,
                confidence_images, intrinsics):
    eps = 1.0 / 90.0
    rgb_out = os.path.join(flags.out, scene_name, 'rgb')
    depth_out = os.path.join(flags.out, scene_name, 'depth')
    pose_out = os.path.join(flags.out, scene_name, 'pose')
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(depth_out, exist_ok=True)
    os.makedirs(pose_out, exist_ok=True)

    images = [(n, p) for n, p in rgb_images.items()]
    images.sort(key=lambda x: to_ts(x[0]))
    for i, (rgb_name, rgb_path_in) in enumerate(images):
        print(f"Writing {rgb_name}", end='\r')
        if rgb_name not in depth_images or rgb_name not in confidence_images:
            print(f"Skipping image {rgb_name}")
            continue

        pose, time_diff = find_pose(trajectory, rgb_name)
        if time_diff > eps:
            print(f"Skipping {rgb_name} due to time diff {time_diff:.03}",
                  end='\r')
            continue
        else:
            print(f"Including {rgb_name} time diff {time_diff:.03}", end='\r')

        T_CW = to_transform(pose)

        image_name = f"{i:06}"
        pose_path = os.path.join(pose_out, image_name + '.txt')
        rgb_path = os.path.join(rgb_out, image_name + '.png')
        depth_path = os.path.join(depth_out, image_name + '.png')

        rgb = cv2.imread(rgb_path_in, -1)
        depth = cv2.imread(depth_images[rgb_name], -1)
        confidence = cv2.imread(confidence_images[rgb_name], -1)
        depth[confidence < 2] = 0
        cv2.imwrite(depth_path, depth)
        cv2.imwrite(rgb_path, rgb)
        np.savetxt(pose_path, T_CW)
    np.savetxt(os.path.join(flags.out, scene_name, 'intrinsics.txt'),
               intrinsics)


def main():
    flags = read_args()

    scenes = os.listdir(flags.arkit_scenes)

    for scene in scenes:
        traj_file = os.path.join(flags.arkit_scenes, scene, 'lowres_wide.traj')
        confidence_dir = os.path.join(flags.arkit_scenes, scene, 'confidence')
        depth_dir = os.path.join(flags.arkit_scenes, scene, 'lowres_depth')
        rgb_dir = os.path.join(flags.arkit_scenes, scene, 'lowres_wide')
        intrinsics_dir = os.path.join(flags.arkit_scenes, scene,
                                      'lowres_wide_intrinsics')

        if not os.path.exists(traj_file) or not os.path.exists(
                confidence_dir) or not os.path.exists(
                    rgb_dir) or not os.path.exists(intrinsics_dir):
            print(f"Missing files in {scene}")
            continue

        trajectory = read_trajectory(traj_file)

        rgb_images = collect_images(rgb_dir)
        depth_images = collect_images(depth_dir)
        confidence_images = collect_images(confidence_dir)
        intrinsics = read_intrinsics(intrinsics_dir)

        write_scene(flags, scene, trajectory, rgb_images, depth_images,
                    confidence_images, intrinsics)


if __name__ == "__main__":
    main()
