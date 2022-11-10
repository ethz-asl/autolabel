

# Mapping with Maplab

This folder contains the source for a Docker image that will run SLAM with Rovioli and [Maplab](https://github.com/ethz-asl/maplab) to compute camera poses and save them in the Autolabel format.

## Building the image

Clone maplab at `ops/maplab/maplab` by running `git clone https://github.com/ethz-asl/maplab.git --recursive -b develop` in this directory.

Build docker image with `docker build -t maplab .`.

To run the mapping pipeline, run:
```
./run.sh map <sensors.yaml> <bag.bag> <out-scene>
```

