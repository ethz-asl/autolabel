import argparse
from argparse import Namespace
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import torch
from torch import optim
import tf
from autolabel.utils.feature_utils import get_feature_extractor
from autolabel.trainer import SimpleTrainer
from autolabel import model_utils
from autolabel.dataset import DynamicDataset
from scipy.spatial.transform import Rotation as R
import threading


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='lseg')
    parser.add_argument('--checkpoint',
                        type=str,
                        default=None,
                        help='path to feature model checkpoint')
    parser.add_argument('--image-topic',
                        type=str,
                        default='/camera/color/image_raw')
    parser.add_argument('--depth-topic',
                        type=str,
                        default='/camera/depth/image_rect_raw')
    return parser.parse_args()


class Frame:

    def __init__(self, T_CW, image, depth, features):
        self.T_CW = T_CW
        self.image = image
        self.depth = depth
        self.features = features


class Bridge:

    def __init__(self, features, checkpoint):
        self.tf_listener = tf.TransformListener()
        self.bridge = CvBridge()
        self.feature_extractor = get_feature_extractor(features, checkpoint)

    def depth_to_array(self, depth_msg):
        depth = self.bridge.imgmsg_to_cv2(depth_msg, 'mono16')
        print("depth", depth.shape)
        return depth / 1000.0

    def color_to_array(self, image_msg):
        return self.bridge.imgmsg_to_cv2(image_msg, 'rgb8')

    def features(self, image_array):
        with torch.inference_mode():
            image = torch.tensor(image_array,
                                 device='cuda:0',
                                 dtype=torch.float16)[None]
            features = self.feature_extractor(image)[0]
        print('computed features:', features.shape)
        return features.cpu().numpy()

    def tf_to_array(self, tf_msg):
        rotation = tf_msg.transforms[0].transform.rotation
        rotation = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
        translation = np.array([
            tf_msg.transforms[0].transform.translation.x,
            tf_msg.transforms[0].transform.translation.y,
            tf_msg.transforms[0].transform.translation.z
        ])
        R_CW = R.from_quat(rotation).as_matrix()
        T_CW = np.eye(4)
        T_CW[:3, :3] = R_CW
        T_CW[:3, 3] = translation
        return T_CW


class TrainingLoop:

    def __init__(self, frame_queue):
        opt = Namespace(rand_pose=-1,
                        color_space='srgb',
                        feature_loss=True,
                        rgb_weight=1.0,
                        depth_weight=0.05,
                        semantic_weight=0.0,
                        feature_weight=0.5)
        #TODO: set these from initial pose.
        min_bounds = np.array([-1., -1., -1.])
        max_bounds = np.array([1., 1., 1.])
        model_options = Namespace(geometric_features=15, feature_dim=512)
        model = model_utils.create_model(min_bounds, max_bounds, 2,
                                         model_options)
        scheduler = lambda optimizer: optim.lr_scheduler.ConstantLR(
            optimizer, 1e-4)
        self.trainer = SimpleTrainer('ngp',
                                     opt,
                                     model,
                                     device='cuda:0',
                                     workspace=None,
                                     fp16=True,
                                     ema_decay=0.95,
                                     lr_scheduler=scheduler)
        self.dataset = DynamicDataset()
        self.initialized = False
        self.training = True
        self.done = True
        self.training_thread = threading.Thread(target=self.train)
        self.training_thread.start()

    def train(self):
        while True:
            if self.done:
                return 0
            if self.initialized and self.training:
                self.trainer.train_iterations(dataset, 1000)

    def add_frame(self, frame):
        self.dataset.add_frame(frame.T_CW, frame.image, frame.depth,
                               frame.features)
        if not self.initialized:
            self.initialized = True

    def stop(self):
        self.training = False
        self.done = True
        self.training_thread.join()


class AutolabelNode:

    def __init__(self, flags):
        self.image_sub = rospy.Subscriber('/slam/rgb', Image,
                                          self.image_callback)
        self.depth_sub = rospy.Subscriber('/slam/depth', Image,
                                          self.depth_callback)
        self.bridge = Bridge(flags.features, flags.checkpoint)
        self.last_depth_msg = None
        self.last_image_msg = None
        self.sync_threshold = 1. / 30.

    def image_callback(self, msg):
        self.last_image_msg = msg
        if np.abs(msg.header.stamp.to_sec() -
                  self.last_depth_msg.header.stamp.to_sec()) < self.sync_threshold:
            self.image_pair(self.last_depth_msg, msg)

    def depth_callback(self, msg):
        self.last_depth_msg = msg
        if np.abs(msg.header.stamp.secs -
                  self.last_image_msg.header.stamp.to_sec()) < self.sync_threshold:
            self.image_pair(msg, self.last_image_msg)

    def image_pair(self, depth_msg, image_msg)
        print(f"Received image pair: {depth_msg.header.stamp.to_sec()} {image_msg.header.stamp.to_sec()}")
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
        tf_msg = self.bridge.tf_listener.lookupTransform(
            'world', image_msg.header.frame_id, image_msg.header.stamp)
        T_CW = self.bridge.tf_to_array(tf_msg)
        image = self.bridge.color_to_array(self.last_image_msg)
        depth = self.bridge.depth_to_array(msg)
        features = self.bridge.features(image)
        frame = Frame(T_CW, image, depth, features)
        self.last_depth_msg = None
        self.last_image_msg = None

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    flags = read_args()
    rospy.init_node("autolabel")
    node = AutolabelNode(flags)
    rospy.spin()
