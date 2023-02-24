from __future__ import print_function
import os
import argparse
import numpy as np
import cv2
import rosbag
import rospy
import skvideo.io
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, Imu
from maplab_msgs.msg import OdometryWithImuBiases

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene')
    parser.add_argument('--out', default='bag.bag', type=str)
    parser.add_argument('--compressed', action='store_true')
    return parser.parse_args()

def read_data(scene):
    intrinsics = np.loadtxt(os.path.join(scene, 'camera_matrix.csv'), delimiter=',')
    odometry = np.loadtxt(os.path.join(scene, 'odometry.csv'), delimiter=',', skiprows=1)
    imu = np.loadtxt(os.path.join(scene, 'imu.csv'), delimiter=',', skiprows=1)
    poses = []

    for line in odometry:
        position = line[2:5]
        quaternion = line[5:]
        T_WC = np.eye(4)
        T_WC[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        T_WC[:3, 3] = position
        poses.append(T_WC)
    depth_dir = os.path.join(scene, 'depth')
    depth_frames = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]
    depth_frames = [f for f in depth_frames if '.npy' in f or '.png' in f]
    return { 'poses': poses, 'intrinsics': intrinsics, 'depth_frames': depth_frames, 'imu': imu, 'odometry': odometry }

def create_pose(t, q):
    T_WC = np.eye(4)
    T_WC[:3, :3] = Rotation.from_quat(q).as_matrix()
    T_WC[:3, 3] = t
    return T_WC

def camera_info_msg(image, timestamp, intrinsics):
    msg = CameraInfo()
    msg.header.stamp = timestamp
    msg.header.frame_id = 'rgb_optical'
    msg.height = image.shape[0]
    msg.width = image.shape[1]
    msg.distortion_model = 'plumb_bob'
    msg.K = intrinsics.ravel().tolist()
    return msg

def main():
    flags = read_args()
    bag = rosbag.Bag(flags.out, 'w')
    data = read_data(flags.scene)

    cv_bridge = CvBridge()

    rgb_topic = '/rgb/image_raw'
    rgb_camera_info_topic = '/rgb/camera_info'
    imu_topic = '/imu'
    pose_topic = '/pose'

    # timestamp, frame, x, y, z, qx, qy, qz, qw
    odometry = data['odometry']
    timestamps = odometry[:, 0]

    # Timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z
    imu = data['imu']
    try:
        reader = skvideo.io.vreader(os.path.join(flags.scene, 'rgb.mp4'))
        for i, (seconds, image) in enumerate(zip(timestamps, reader)):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if flags.compressed:
                msg = cv_bridge.cv2_to_compressed_imgmsg(image)
            else:
                msg = cv_bridge.cv2_to_imgmsg(image)
            print('Writing image {i} timestamp {seconds:.02f}'.format(i=i, seconds=seconds), end='\r')
            ts = rospy.Time(seconds)
            msg.header.stamp = ts
            msg.header.seq = i
            msg.header.frame_id = 'rgb_optical'
            bag.write(rgb_topic, msg, t=ts)

            info_msg = camera_info_msg(image, ts, data['intrinsics'])
            bag.write(rgb_camera_info_topic, info_msg, t=ts)

        for i, reading in enumerate(imu):
            seconds = reading[0]
            distance = np.abs(timestamps - seconds).min()
            print('Writing imu message timestamp {seconds:.02f} closest frame {distance:.03f}'.format(seconds=seconds, distance=distance), end='\r')
            msg = Imu()
            ts = rospy.Time(seconds)
            msg.header.stamp = ts
            msg.header.seq = i
            msg.header.frame_id = 'imu_frame'
            msg.linear_acceleration.x = reading[1]
            msg.linear_acceleration.y = reading[2]
            msg.linear_acceleration.z = reading[3]
            msg.angular_velocity.x = reading[4]
            msg.angular_velocity.y = reading[5]
            msg.angular_velocity.z = reading[6]

            bag.write(imu_topic, msg, t=ts)

        for i, reading in enumerate(odometry):
            ts = rospy.Time(reading[0])
            msg = OdometryWithImuBiases()
            msg.header.stamp = ts
            msg.header.seq = i
            msg.header.frame_id = 'map'
            msg.child_frame_id = 'rgb_optical'
            T_WC = create_pose(reading[2:5], reading[5:])
            # T_CW = np.linalg.inv(T_WC)
            translation = T_WC[:3, 3]
            orn = Rotation.from_matrix(T_WC[:3, :3]).as_quat()
            msg.pose.pose.position.x = translation[0]
            msg.pose.pose.position.y = translation[1]
            msg.pose.pose.position.z = translation[2]
            msg.pose.pose.orientation.x = orn[0]
            msg.pose.pose.orientation.y = orn[1]
            msg.pose.pose.orientation.z = orn[2]
            msg.pose.pose.orientation.w = orn[3]
            bag.write(pose_topic, msg, t=ts)

    finally:
        bag.close()



if __name__ == "__main__":
    main()
