import argparse
import os
import time
from argparse import Namespace
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
import tf
from autolabel.utils.feature_utils import get_feature_extractor
from autolabel.utils import Camera
from autolabel.trainer import SimpleTrainer
from autolabel import model_utils
from autolabel.dataset import DynamicDataset
from autolabel.dataset import _compute_direction
from scipy.spatial.transform import Rotation
import threading


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='lseg')
    parser.add_argument('--checkpoint',
                        type=str,
                        default=None,
                        help='path to feature model checkpoint')
    parser.add_argument(
        '--log',
        default=None,
        type=str,
        help=
        "Save incoming images to this directory in the autolabel format for debugging."
    )
    return parser.parse_args()


def to_pose(pose_stamped):
    T_CW = np.eye(4)
    T_CW[:3, :3] = Rotation.from_quat([
        pose_stamped.pose.orientation.x, pose_stamped.pose.orientation.y,
        pose_stamped.pose.orientation.z, pose_stamped.pose.orientation.w
    ]).as_matrix()
    T_CW[:3, 3] = [
        pose_stamped.pose.position.x, pose_stamped.pose.position.y,
        pose_stamped.pose.position.z
    ]
    return T_CW.astype(np.float32)


class Frame:

    def __init__(self, num, T_CW, image, depth, features):
        self.num = num
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
        image = np.transpose(image_array / 255., [2, 0, 1])
        with torch.inference_mode():
            image = torch.tensor(image, device='cuda:0',
                                 dtype=torch.float32)[None]
            features = self.feature_extractor(image)[0]
            features = F.normalize(features, dim=0)
        return features.cpu().numpy()

    def tf_to_array(self, translation, orientation):
        R_CW = Rotation.from_quat(orientation).as_matrix()
        T_CW = np.eye(4)
        T_CW[:3, :3] = R_CW
        T_CW[:3, 3] = translation
        return T_CW

    def image_to_message(self, array):
        msg = self.bridge.cv2_to_imgmsg(array, encoding='rgb8')
        msg.header.stamp = rospy.Time.now()
        return msg


class TrainingLoop:

    def __init__(self, bridge):
        self.bridge = bridge
        min_bounds = np.array([-2.5, -2.5, -2.5])
        max_bounds = np.array([2.5, 2.5, 2.5])
        lr = 1e-4
        # scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer,
        #         [10, 25, 50], gamma=0.1)
        scheduler = lambda optimizer: optim.lr_scheduler.ConstantLR(
            optimizer, lr)
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
        ],
                                                   lr=lr,
                                                   betas=(0.9, 0.99),
                                                   eps=1e-15)
        opt = Namespace(rand_pose=-1,
                        color_space='srgb',
                        feature_loss=True,
                        encoding='hg+freq',
                        rgb_weight=1.0,
                        geometric_features=15,
                        feature_dim=512,
                        depth_weight=0.1,
                        semantic_weight=0.0,
                        feature_weight=0.0)
        self.model = model_utils.create_model(min_bounds, max_bounds, 2, opt)
        self.trainer = SimpleTrainer(
            'ngp',
            opt,
            self.model,
            criterion=torch.nn.MSELoss(reduction='none'),
            optimizer=optimizer,
            device='cuda:0',
            workspace=None,
            fp16=True,
            ema_decay=0.95,
            lr_scheduler=scheduler)
        self.camera = Camera(
            np.array([[513.104, 0.0, 321.532], [0.0, 513.104, 184.124],
                      [0., 0., 1.]]), (640, 360))
        self.dataset = DynamicDataset(4096, self.camera)
        self.loader = torch.utils.data.DataLoader(self.dataset,
                                                  batch_size=None,
                                                  num_workers=0)
        self.initialized = False
        self.training = True
        self.done = False
        self.training_thread = threading.Thread(target=self.train)
        self.training_thread.start()
        self.render_resolution = (256, 192)
        self.pixel_indices = np.arange(self.render_resolution[0] *
                                       self.render_resolution[1])
        self.odometry_pose = None
        self.image_pub = rospy.Publisher('/autolabel/image',
                                         Image,
                                         queue_size=1)
        self.feature_pub = rospy.Publisher('/autolabel/features',
                                           Image,
                                           queue_size=1)

    def train(self):
        while True:
            if self.done:
                print("Closing training loop")
                return 0
            if self.initialized and self.training and len(self.dataset) > 5:
                print(f"Fitting with {len(self.dataset)} images")
                self.model.train()
                self.trainer.train_iterations(self.loader, 100)
                if self.odometry_pose is not None:
                    self.model.eval()
                    self.render_frame()
            else:
                time.sleep(0.05)

    def render_frame(self):
        T_CW = self.odometry_pose
        resolution = self.render_resolution
        T_WC = self.dataset._convert_pose(T_CW)
        origin = T_WC[:3, 3]
        origins = np.broadcast_to(
            origin, (resolution[1], resolution[0], 3)).astype(np.float32)
        R_WC = np.ascontiguousarray(T_WC[:3, :3])
        fx = 205.
        fy = 205.
        cx = 256. / 2.
        cy = 192. / 2.
        directions = _compute_direction(R_WC, self.pixel_indices, resolution[0],
                                        fx, fy, cx, cy, False)
        directions = directions.reshape(
            (self.render_resolution[1], self.render_resolution[0], 3))
        rays_o = torch.tensor(origins, device='cuda:0', dtype=torch.float16)
        rays_d = torch.tensor(directions, device='cuda:0', dtype=torch.float16)
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model.render(rays_o,
                                            rays_d,
                                            staged=True,
                                            perturb=False,
                                            max_ray_batch=2048)
        image = outputs['image']
        features = outputs['semantic_features']
        image = (image * 255.).cpu().numpy().astype(np.uint8)
        msg = self.bridge.image_to_message(image)
        self.image_pub.publish(msg)
        print("image:", image.shape)
        print("features:", features.shape)

    def add_frame(self, frame):
        print("received frame")
        self.dataset.add_frame(frame.T_CW, frame.image, frame.depth,
                               frame.features)
        self.initialized = True

    def stop(self):
        self.training = False
        self.done = True
        self.training_thread.join()
        self.dataset.stop()


class AutolabelNode:

    def __init__(self, flags):
        self.bridge = Bridge(flags.features, flags.checkpoint)
        self.sync_threshold = 1. / 30.
        self.training_loop = TrainingLoop(self.bridge)
        self.image_sub = rospy.Subscriber('/slam/rgb',
                                          Image,
                                          self.image_callback,
                                          queue_size=20)
        self.depth_sub = rospy.Subscriber('/slam/depth',
                                          Image,
                                          self.depth_callback,
                                          queue_size=20)
        self.odometry_sub = rospy.Subscriber('/slam/odometry', PoseStamped,
                                             self.odometry_callback)
        self.keyframe_sub = rospy.Subscriber('/slam/keyframe',
                                             PoseStamped,
                                             self.keyframe_callback,
                                             queue_size=20)
        self.depth_messages = {}
        self.rgb_messages = {}
        self.pose_messages = {}
        self.debug_log = flags.log
        if self.debug_log is not None:
            os.makedirs(os.path.join(self.debug_log, 'rgb'), exist_ok=True)
            os.makedirs(os.path.join(self.debug_log, 'depth'), exist_ok=True)
            os.makedirs(os.path.join(self.debug_log, 'pose'), exist_ok=True)
            self.training_loop.camera.save(
                os.path.join(self.debug_log, 'intrinsics.txt'))

    def image_callback(self, msg):
        self.rgb_messages[msg.header.seq] = msg
        self._check_tuple(msg.header.seq)

    def depth_callback(self, msg):
        self.depth_messages[msg.header.seq] = msg
        self._check_tuple(msg.header.seq)

    def keyframe_callback(self, msg):
        self.pose_messages[msg.header.seq] = msg
        self._check_tuple(msg.header.seq)

    def _check_tuple(self, seq_num):
        if (seq_num in self.depth_messages and seq_num in self.rgb_messages and
                seq_num in self.pose_messages):
            rgb = self.rgb_messages[seq_num]
            depth = self.depth_messages[seq_num]
            pose = self.pose_messages[seq_num]
            self.image_tuple(seq_num, rgb, depth, pose)
            del self.rgb_messages[seq_num]
            del self.depth_messages[seq_num]
            del self.pose_messages[seq_num]

    def image_tuple(self, num, depth_msg, image_msg, pose_msg):
        T_CW = to_pose(pose_msg)
        if np.abs(depth_msg.header.stamp.to_sec() -
                  image_msg.header.stamp.to_sec()) > self.sync_threshold:
            print("WARNING depth and rgb might not be synchronized")
        image = self.bridge.color_to_array(image_msg)
        depth = self.bridge.depth_to_array(depth_msg)
        features = self.bridge.features(image)
        frame = Frame(num, T_CW, image, depth, features)
        self.training_loop.add_frame(frame)
        if self.debug_log is not None:
            self._debug_log_frame(frame)

    def _debug_log_frame(self, frame):
        filename = f"{frame.num:06d}"
        cv2.imwrite(os.path.join(self.debug_log, 'rgb', f"{filename}.jpg"),
                    cv2.cvtColor(frame.image, cv2.COLOR_RGB2RGB))
        cv2.imwrite(os.path.join(self.debug_log, 'depth', f"{filename}.png"),
                    frame.depth)
        np.savetxt(os.path.join(self.debug_log, 'pose', f"{filename}.txt"),
                   frame.T_CW)

    def odometry_callback(self, msg):
        self.training_loop.odometry_pose = to_pose(msg)

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
