

# Mapping with Maplab

This folder contains the source for a Docker image that will run SLAM with Rovioli and [Maplab](https://github.com/ethz-asl/maplab) to compute camera poses and save them in the Autolabel format.

## Building the image

Clone maplab at `ops/maplab/maplab` by running `git clone https://github.com/ethz-asl/maplab.git --recursive -b develop` in this directory.

Build docker image with `docker build -t maplab .`.

## Run the docker image

To run the mapping pipeline, run:
```
./run.sh map <sensors.yaml> <bag.bag> <out-scene>
```

NOTE: at the moment, some of the topic names are hard coded for our Azure Kinect setup. See the example sensor configuration at `sensors.yaml`.

The `run.sh` script also has a shell mode, which lets you run a bash shell inside the docker container for debugging purposes.
```
./run.sh shell <sensors.yaml> <bag.bag> <out-scene>
```

## RViz

While the pipeline is running, you can visualize the map in RViz using [this](https://github.com/ethz-asl/maplab/blob/pre_release_public/july-2018/applications/rovioli/share/rviz-rovioli.rviz) RViz config.

