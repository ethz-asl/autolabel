#!/bin/bash

set -e

absolute_path() {
	echo "$(readlink -m $1)"
}

USAGE="Usage: ./run.sh <command>"
USAGE+="\nmap <sensors.yaml> <map-folder> <ros.bag>"
USAGE+="\nshell"
command="$1"

if [ "$command" == "shell" ];
then
	sensor_file="$(absolute_path $2)"
	data_dir="$(absolute_path $3)"
	out_scene="$4"

	mkdir -p $out_scene

	xhost +local:root;
	docker run -it --runtime=nvidia --privileged -v /dev/bus/usb:/dev/bus/usb \
		--entrypoint /bin/bash \
		-v "$sensor_file":/home/maplab_user/sensors.yaml \
		-v "$data_dir":/home/maplab_user/data \
		-v "$out_scene":/home/maplab_user/out_scene \
		-v "/home/ken/Hack/autolabel/ops/maplab":/home/maplab_user/src \
		-e QT_X11_NO_MITSHM=1 --network=host \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		maplab
	xhost +local:root;
elif [ "$command" == "map" ];
then
	sensor_file="$(absolute_path $2)"
	bag_file="$(absolute_path $3)"
	out_scene="$4"

	if [ "$sensor_file" == "" ];
	then
		echo "Sensor file required. "
		echo "$USAGE"
		exit 1
	fi
	if [ "$bag_file" == "" ];
	then
		echo "Bag file required."
		echo "$USAGE"
		exit 1
	fi
	if [ "$out_scene" == "" ];
	then
		echo "Out scene folder required."
		echo "$USAGE"
		exit 1
	fi

	mkdir -p $out_scene

	xhost +local:root;
	docker run -it --runtime=nvidia --privileged -v /dev/bus/usb:/dev/bus/usb \
		-v "$sensor_file":/home/maplab_user/sensors.yaml \
		-v "$bag_file":/home/maplab_user/bag.bag \
		-v "$out_scene":/home/maplab_user/out_scene \
		-e QT_X11_NO_MITSHM=1 --network=host \
		-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		maplab
	xhost +local:root;
else
	echo "Command $1 not recognized. Try map or shell."
fi
