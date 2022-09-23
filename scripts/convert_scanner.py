import argparse
import os
import numpy as np
import json
import cv2
from skvideo import io


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scan', type=str)
    parser.add_argument('--out', type=str)
    return parser.parse_args()


def write_frames(flags, rgb_out_dir):
    rgb_video = os.path.join(flags.scan, 'rgb.mp4')
    video = io.vreader(rgb_video)
    for i, frame in enumerate(video):
        print(f"Writing rgb frame {i:05}" + " " * 10, end='\r')
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_path = os.path.join(rgb_out_dir, f"{i:05}.jpg")
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        cv2.imwrite(frame_path, frame, params)


def write_depth(flags, depth_out_dir):
    depth_dir_in = os.path.join(flags.scan, 'depth')
    confidence_dir = os.path.join(flags.scan, 'confidence')
    files = sorted(os.listdir(depth_dir_in))
    for filename in files:
        if '.png' not in filename:
            continue
        print(f"Writing depth frame {filename}", end='\r')
        number, _ = filename.split('.')
        depth = cv2.imread(os.path.join(depth_dir_in, filename), -1)
        confidence = cv2.imread(os.path.join(confidence_dir,
                                             number + '.png'))[:, :, 0]
        depth[confidence < 2] = 0
        cv2.imwrite(os.path.join(depth_out_dir, f"{int(number):05}" + '.png'),
                    depth)


def write_intrinsics(flags):
    intrinsics = np.loadtxt(os.path.join(flags.scan, 'camera_matrix.csv'),
                            delimiter=',')
    np.savetxt(os.path.join(flags.out, 'intrinsics.txt'), intrinsics)


def main():
    flags = read_args()
    rgb_out = os.path.join(flags.out, 'raw_rgb/')
    depth_out = os.path.join(flags.out, 'raw_depth/')
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(depth_out, exist_ok=True)

    write_intrinsics(flags)
    write_depth(flags, depth_out)
    write_frames(flags, rgb_out)
    print("\nDone.")


if __name__ == "__main__":
    main()
