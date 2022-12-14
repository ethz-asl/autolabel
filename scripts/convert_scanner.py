import argparse
import os
import numpy as np
import json
import cv2
from skvideo import io
from tqdm import tqdm


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scan', type=str, help="Scan directory")
    parser.add_argument('--out', type=str, help="Output directory")
    parser.add_argument("--rotate",
                        action="store_true",
                        help="Rotate frames 90 degrees")

    parser.add_argument("--subsample",
                        type=int,
                        default=1,
                        help="Subsample use every n frames from the dataset")
    return parser.parse_args()


def write_frames(scan_dir, rgb_out_dir, rotate=False, subsample=1):
    rgb_video = os.path.join(scan_dir, 'rgb.mp4')
    video = io.vreader(rgb_video)
    img_idx = 0
    for i, frame in tqdm(enumerate(video), desc="Writing RGB"):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if i % subsample != 0:
            continue
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame_path = os.path.join(rgb_out_dir, f"{img_idx:05}.jpg")
        img_idx += 1
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        cv2.imwrite(frame_path, frame, params)


def write_depth(scan_dir, depth_out_dir, rotate=False, subsample=1):
    depth_dir_in = os.path.join(scan_dir, 'depth')
    confidence_dir = os.path.join(scan_dir, 'confidence')
    files = sorted(os.listdir(depth_dir_in))
    img_idx = 0
    for i, filename in tqdm(enumerate(files), desc="Writing Depth"):
        if '.png' not in filename:
            continue
        number, _ = filename.split('.')

        if i % subsample != 0:
            continue

        depth = cv2.imread(os.path.join(depth_dir_in, filename), -1)

        confidence = cv2.imread(os.path.join(confidence_dir,
                                             number + '.png'))[:, :, 0]
        if rotate:
            depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
            confidence = cv2.rotate(confidence, cv2.ROTATE_90_CLOCKWISE)

        depth[confidence < 2] = 0
        cv2.imwrite(os.path.join(depth_out_dir, f"{int(img_idx):05}" + '.png'),
                    depth)
        img_idx += 1
    return img_idx


def write_intrinsics(scan_dir, out_dir, rotate=False):
    intrinsics = np.loadtxt(os.path.join(scan_dir, 'camera_matrix.csv'),
                            delimiter=',')
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    if rotate:
        out_intrinsics = np.array([[fy, 0, cy], [0, fx, cx], [0, 0, 1]])
    else:
        out_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    np.savetxt(os.path.join(out_dir, 'intrinsics.txt'), out_intrinsics)


def main():
    flags = read_args()
    rgb_out = os.path.join(flags.out, 'raw_rgb/')
    depth_out = os.path.join(flags.out, 'raw_depth/')
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(depth_out, exist_ok=True)
    scan_dir = flags.scan

    write_intrinsics(scan_dir, flags.out, rotate=flags.rotate)
    write_depth(scan_dir,
                depth_out,
                rotate=flags.rotate,
                subsample=flags.subsample)
    write_frames(scan_dir,
                 rgb_out,
                 rotate=flags.rotate,
                 subsample=flags.subsample)
    print("Done")


if __name__ == "__main__":
    main()
