# Vision-language feature fields


## Installing LSeg

In addition to the regular installation instructions, you also need to install LSeg. This can be done by running the following commands with your python environment loaded.
```
git clone https://github.com/kekeblom/lang-seg
cd lang-seg
pip install -e .
```

## Running ScanNet experiment

First follow the instructions in `docs/data.md` to convert the ScanNet scenes into the right format. Then use the following commands to compute vision-language features, fit the scene representation and evaluate against the ground truth:
```
# Train . Has to be run separately for each scene.
python scripts/compute_feature_maps.py <dataset-dir>/<scene> --features lseg --checkpoint <lseg-weights>
python scripts/train.py --features lseg --feature-dim 512 --iters 25000 <dataset-dir>/<scene>

# Once trained on all scenes, evaluate.
# 3D queries evaluated against the 3D pointcloud
python scripts/language/evaluate.py --pc --label-map <label-map> --feature-checkpoint <lseg-weights> <dataset-dir>
# 2D queries against the ground truth semantic segmentation maps
python scripts/language/evaluate.py --label-map <label-map> --feature-checkpoint <lseg-weights> <dataset-dir>
```

`dataset-dir` is the path to the scannet converted scenes, `scene` is the name of the scene. `lseg-weights` is the path to the lseg checkpoint `.ckpt` file which contain the lseg model weights.

## Running the real-time ROS node

The `scripts/ros/` directory contains ROS nodes which can be used to integrate with a real-time SLAM system. These have been tested under ROS Noetic.

`scripts/ros/node.py` is the node which listens to keyframes and integrates the volumetric representation as they come in. It listens to the following topics:
- `/slam/rgb` image messages.
- `/slam/depth` depth frames encoded as uint16 values in millimeters.
- `/slam/keyframe` PoseStamped messages which correspond to camera poses for the rgb and depth messages.
- `/slam/camera_info` CameraInfo message containing the intrinsic parameters.
- `/slam/odometry` (optional) PoseStamped messages. Each time a message comes in, it renders an rgb frame and semantic segmentation map which is published at `/autolabel/image` and `/autolabel/features` respectively.
- `/autolabel/segmentation_classes` segmentation class prompts as a String message published by the `class_input.py` node.

It can be run with `python scripts/ros/node.py --checkpoint <lseg-weights> -b <bound>`. The bound parameter is optional and defaults to 2.5 meters. It defines the size of the volume, extending `bound` meters from `[-bound, -bound, -bound]` to `[bound, bound, bound]` in the x, y and z directions.

For an implementation of the SLAM node, you can use the ROS node from the [SpectacularAI SDK examples](https://github.com/SpectacularAI/sdk-examples/blob/main/python/oak/mapping_ros.py), in case you have an OAK-D stereo camera.

`scripts/ros/class_input.py` presents a graphical user interface which can be used to define the segmentation classes used by the ROS node. It published class at `/autolabel/segmentation_classes`.

