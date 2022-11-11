from __future__ import print_function
import argparse
import numpy as np
import os
import rosbag
import csv
import cv2
import yaml
import open3d as o3d
from scipy.spatial.transform import Rotation


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag',
                        required=True,
                        help="Path to bag file that was mapped.")
    parser.add_argument('--export',
                        default="/tmp/maps/csv_export.csv",
                        help="Path to maplab csv export.")
    parser.add_argument('--out',
                        required=True,
                        help="Where to write the resulting scene.")
    parser.add_argument('--sensors',
                        required=True,
                        help="Maplab sensor config.")
    return parser.parse_args()


def read_csv(filepath):
    array = np.loadtxt(filepath)
    return array[:, 0], array


class Frame:

    def __init__(self, t_img):
        self.t_img = t_img
        self.t_imu = None
        self.t_depth = None
        self.T_CW = None
        self.image = None
        self.depth = None


class BBoxComputer:

    def __init__(self, K, image_size):
        self.min_bounds = np.zeros(3)
        self.max_bounds = np.zeros(3)
        self.K = o3d.camera.PinholeCameraIntrinsic(int(image_size[0]),
                                                   int(image_size[1]), K[0, 0],
                                                   K[1, 1], K[0, 2], K[1, 2])
        self.pc = o3d.geometry.PointCloud()

    def add_frame(self, T_CW, depth):
        depth = o3d.geometry.Image(depth)
        pc_C = o3d.geometry.PointCloud.create_from_depth_image(
            depth, depth_scale=1000.0, intrinsic=self.K)
        pc_C = np.asarray(pc_C.points)
        T_WC = np.linalg.inv(T_CW)
        pc_W = (T_WC[:3, :3].__matmul__(pc_C[:, :, None]))[:, :, 0] + T_WC[:3,
                                                                           3]

        self.min_bounds = np.minimum(self.min_bounds, pc_W.min(axis=0))
        self.max_bounds = np.maximum(self.max_bounds, pc_W.max(axis=0))
        self.pc += o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(pc_W)).uniform_down_sample(50)

    def get_bounds(self):
        filtered, _ = self.pc.remove_statistical_outlier(nb_neighbors=20,
                                                         std_ratio=2.0)
        bbox = filtered.get_oriented_bounding_box()
        T = np.eye(4)
        T[:3, :3] = bbox.R.T
        o3d_aabb = o3d.geometry.PointCloud(filtered).transform(
            T).get_axis_aligned_bounding_box()
        center = o3d_aabb.get_center()
        T[:3, 3] = -center
        aabb = np.zeros((2, 3))
        aabb[0, :] = o3d_aabb.get_min_bound() - center
        aabb[1, :] = o3d_aabb.get_max_bound() - center
        return T, aabb, filtered


def to_pose(vertex):
    t = vertex[1:4]
    q = vertex[4:]
    T_WI = np.eye(4)
    T_WI[:3, 3] = t
    R_CW = Rotation.from_quat(q)
    T_WI[:3, :3] = R_CW.as_matrix()
    return np.linalg.inv(T_WI)


def collect_frames(bag, timestamps, vertices, sensor_filepath):
    frames = []
    T_CI = np.eye(4)

    with open(sensor_filepath, 'rt') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for sensor in config['sensors']:
        if sensor.get('sensor_type') == 'NCAMERA':
            camera = sensor['cameras'][0]['T_B_C']
            break
    T_IC = np.array(camera['data']).reshape(4, 4)
    T_CI = np.linalg.inv(T_IC)

    #TODO: Lookup the topic names from somehwere.
    for topic, msg, t in bag.read_messages(topics="/rgb/image_rect_color"):
        closest = np.abs(timestamps - msg.header.stamp.to_sec()).argmin()
        distance_to_closest = timestamps[closest] - msg.header.stamp.to_sec()
        if distance_to_closest > 0.05:
            print(
                "Frame at time {} is too far away from a measurement with distance of {} seconds."
                .format(msg.header.stamp.to_sec(), distance_to_closest))
        else:
            frame = Frame(msg.header.stamp.to_sec())
            frame.image = msg
            frame.t_imu = timestamps[closest]
            T_IW = to_pose(vertices[closest])
            T_CW = T_CI.__matmul__(T_IW)
            frame.T_CW = T_CW
            frames.append(frame)

    frame_times = np.array([t.t_img for t in frames])
    for topic, msg, t in bag.read_messages(topics="/depth_to_rgb/image_rect"):
        closest_img = np.abs(frame_times - msg.header.stamp.to_sec()).argmin()
        frame = frames[closest_img]
        if frame.depth is not None:
            print("Found two rgb images to match depth.")
        frame.depth = msg
        frame.t_depth = msg.header.stamp.to_sec()

    without_depth = [f for f in frames if f.depth is None]
    if len(without_depth) > 0:
        print("Skipping {} frames without depth frame.".format(
            len(without_depth)))

    frames = [f for f in frames if f.depth is not None]
    return frames


def get_intrinsics(bag):
    for topic, msg, t in bag.read_messages(topics='/rgb/camera_info'):
        return msg


def write_scene(out_dir, frames, intrinsics):
    rgb_out = os.path.join(out_dir, 'rgb')
    depth_out = os.path.join(out_dir, 'depth')
    pose_out = os.path.join(out_dir, 'pose')
    intrinsics_file = os.path.join(out_dir, 'intrinsics.txt')
    distortion_file = os.path.join(out_dir, 'distortion_parameters.txt')

    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(depth_out, exist_ok=True)
    os.makedirs(pose_out, exist_ok=True)

    K = np.array(intrinsics.K).reshape(3, 3)
    np.savetxt(intrinsics_file, K)

    frames = sorted(frames, key=lambda f: f.t_img)

    bbox_computer = None
    for i, frame in enumerate(frames):
        rgb = np.frombuffer(frame.image.data,
                            dtype=np.uint8).reshape(frame.image.height,
                                                    frame.image.width, -1)
        if cv2.waitKey(1) == ord('q'):
            break
        depth = np.frombuffer(frame.depth.data,
                              dtype=np.uint16).reshape(frame.depth.height,
                                                       frame.depth.width)
        if bbox_computer is None:
            bbox_computer = BBoxComputer(K, (depth.shape[1], depth.shape[0]))
        if i % 5 == 0:
            # No need to add every single one.
            bbox_computer.add_frame(frame.T_CW, depth.astype(np.uint16))
        assert depth.dtype == np.uint16

        frame_name = "{i:05}".format(i=i)
        rgb_name = "{}.jpg".format(frame_name)
        depth_name = "{}.png".format(frame_name)
        rgb_path = os.path.join(rgb_out, rgb_name)
        depth_path = os.path.join(depth_out, depth_name)
        cv2.imwrite(rgb_path, rgb)
        cv2.imwrite(depth_path, depth)

    T, bounds, _ = bbox_computer.get_bounds()
    for i, frame in enumerate(frames):
        frame_name = "{i:05}".format(i=i)
        pose_path = os.path.join(pose_out, "{}.txt".format(frame_name))
        T_CW = frame.T_CW
        # Apply transform to align with computed bounding box.
        T_WC = T.__matmul__(np.linalg.inv(T_CW))
        np.savetxt(pose_path, np.linalg.inv(T_WC))

    bbox_path = os.path.join(out_dir, 'bbox.txt')
    with open(bbox_path, 'wt') as f:
        min_str = " ".join([str(x) for x in bounds[0]])
        max_str = " ".join([str(x) for x in bounds[1]])
        f.write("{} {} 0.01".format(min_str, max_str))


def main():
    flags = read_args()

    vertices_file = flags.export
    timestamps, vertices = read_csv(vertices_file)

    bag = rosbag.Bag(flags.bag, 'r')

    frames = collect_frames(bag, timestamps, vertices, flags.sensors)
    intrinsics = get_intrinsics(bag)

    write_scene(flags.out, frames, intrinsics)
    print("Done")


if __name__ == "__main__":
    main()
