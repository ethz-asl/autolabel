#!/bin/bash

set -e

ROSBAG=/home/maplab_user/bag.bag
MAP_FOLDER=/tmp/maps/map
SENSOR_CALIBRATION=/home/maplab_user/sensors.yaml

bash $HOME/run_rovioli.sh

python3.7 $HOME/convert_to_autolabel.py --bag $ROSBAG --out $HOME/out_scene --sensors $SENSOR_CALIBRATION

