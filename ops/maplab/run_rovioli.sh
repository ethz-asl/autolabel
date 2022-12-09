#!/bin/bash

set -e

ROSBAG=/home/maplab_user/bag.bag
MAP_FOLDER=/tmp/maps/map
SENSOR_CALIBRATION=/home/maplab_user/sensors.yaml

source /home/maplab_user/ws/devel/setup.bash

rosrun rovioli rovioli \
	  --v=1 \
		--alsologtostderr=1 \
		--vio_camera_topic_suffix "image_raw" \
		--sensor_calibration_file "$SENSOR_CALIBRATION" \
		--datasource_type "rosbag" \
		--save_map_folder "$MAP_FOLDER" \
		--optimize_map_to_localization_map false \
		--map_builder_save_image_as_resources false \
		--datasource_rosbag $ROSBAG \
		--vio_rosbag_realtime_playback_rate=2.0 \
		--vio_nframe_sync_max_output_frequency_hz 30 \
		--overwrite=true \
		--imu_to_camera_time_offset_ns "20871832" \
		--rovio_enable_frame_visualization=true \
		--rovioli_visualize_map \
		--map_builder_save_image_as_resources=false \
		--feature_tracker_visualize_keypoint_matches

rosrun maplab_console batch_runner --batch_control_file $HOME/maplab_console_script.yaml

