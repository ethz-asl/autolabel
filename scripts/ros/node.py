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
from autolabel.utils import Camera
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
        return self.bridge.imgmsg_to_cv2(depth_msg, 'mono16')

    def color_to_array(self, image_msg):
        return self.bridge.imgmsg_to_cv2(image_msg, 'rgb8')

    def features(self, image_array):
        """
        image_array: H x W x 3 rgb image
        returns: H_o x W_o x D image features
        """
        with torch.inference_mode():
            image = torch.tensor(np.transpose(image_array, [2, 0, 1]),
                                 device='cuda:0',
                                 dtype=torch.float16)[None]
            features = self.feature_extractor(image)[0]
        print('computed features:', features.shape)
        return features.cpu().numpy()

    def tf_to_array(self, translation, orientation):
        R_CW = R.from_quat(orientation).as_matrix()
        T_CW = np.eye(4)
        T_CW[:3, :3] = R_CW
        T_CW[:3, 3] = translation
        return T_CW


class TrainingLoop:

    def __init__(self):
        #TODO: set these from initial pose.
        min_bounds = np.array([-5., -5., -5.])
        max_bounds = np.array([5., 5., 5.])
        scheduler = lambda optimizer: optim.lr_scheduler.ConstantLR(
            optimizer, 1e-3)
        optimizer = lambda model: torch.optim.Adam([
            {
                'name': 'encoding',
                'params': list(model.encoder.parameters())
            },
            {
                'name': 'net',
                'params': model.network_parameters(),
                'weight_decay': 1e-6
            },
        ], lr=1e-4, betas=(0.9, 0.99), eps=1e-15)
        opt = Namespace(rand_pose=-1,
                        color_space='srgb',
                        feature_loss=True,
                        encoding='hg+freq',
                        rgb_weight=1.0,
                        geometric_features=15,
                        feature_dim=512,
                        depth_weight=0.05,
                        semantic_weight=0.0,
                        feature_weight=0.5)
        model = model_utils.create_model(min_bounds, max_bounds, 2,
                                         opt)
        self.trainer = SimpleTrainer('ngp',
                                     opt,
                                     model,
                                     criterion=torch.nn.MSELoss(reduction='none'),
                                     optimizer=optimizer,
                                     device='cuda:0',
                                     workspace=None,
                                     fp16=True,
                                     ema_decay=0.95,
                                     lr_scheduler=scheduler)
        camera = Camera(np.array([[513.104, 0.0, 321.532],
            [0.0, 513.104, 104.124],
            [0., 0., 1.]]), (640, 360))
        self.dataset = DynamicDataset(2048, camera)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=None, num_workers=0)
        self.initialized = False
        self.training = True
        self.done = False
        self.training_thread = threading.Thread(target=self.train)
        self.training_thread.start()

    def train(self):
        while True:
            if self.done:
                print("Closing training loop")
                return 0
            if self.initialized and self.training:
                print(f"Fitting with {len(self.dataset)} images")
                self.trainer.train_iterations(self.loader, 100)

    def add_frame(self, frame):
        print("received frame")
        self.dataset.add_frame(frame.T_CW, frame.image, frame.depth,
                               frame.features)
        self.initialized = True

    def stop(self):
        self.training = False
        self.done = True
        self.training_thread.join()


class AutolabelNode:

    def __init__(self, flags):
        self.bridge = Bridge(flags.features, flags.checkpoint)
        self.last_depth_msg = None
        self.last_image_msg = None
        self.sync_threshold = 1. / 30.
        self.training_loop = TrainingLoop()
        self.image_sub = rospy.Subscriber('/slam/rgb', Image,
                                          self.image_callback)
        self.depth_sub = rospy.Subscriber('/slam/depth', Image,
                                          self.depth_callback)

    def image_callback(self, msg):
        self.last_image_msg = msg
        if self._has_pair():
            self.image_pair(self.last_depth_msg, msg)

    def depth_callback(self, msg):
        self.last_depth_msg = msg
        if self._has_pair():
            self.image_pair(msg, self.last_image_msg)

    def _has_pair(self):
        return (self.last_image_msg is not None and
                self.last_depth_msg is not None and
                np.abs(self.last_image_msg.header.stamp.to_sec() - self.last_depth_msg.header.stamp.to_sec()) < self.sync_threshold)

    def image_pair(self, depth_msg, image_msg):
        print(f"Received image pair: {depth_msg.header.stamp.to_sec()} {image_msg.header.stamp.to_sec()}")
        try:
            self.bridge.tf_listener.waitForTransform('world',
                    image_msg.header.frame_id,
                    image_msg.header.stamp,
                    rospy.Duration(10.))
            tf_msg = self.bridge.tf_listener.lookupTransform(
                'world', image_msg.header.frame_id, image_msg.header.stamp)
        except tf.ExtrapolationException:
            print("Can't lookup transform")
            return
        T_CW = self.bridge.tf_to_array(*tf_msg)
        image = self.bridge.color_to_array(self.last_image_msg)
        depth = self.bridge.depth_to_array(self.last_depth_msg)
        features = self.bridge.features(image)
        frame = Frame(T_CW, image, depth, features)
        self.training_loop.add_frame(frame)
        self.last_depth_msg = None
        self.last_image_msg = None

    def run(self):
        rospy.spin()

    def stop(self):
        self.training_loop.stop()


if __name__ == "__main__":
    flags = read_args()
    rospy.init_node("autolabel")
    try:
        node = AutolabelNode(flags)
        node.run()
    finally:
        node.stop()

