# Autolabel

This goal of this project is to provide accurate ground truth data for RGB-D sensor stream.

## User interface

The main way to interact with this project is through the graphical user interface.

Run the user interface with `python scripts/gui.py <scene>`

## Scene directory structure

The scene directory structure is as follows:
```
rgb/            # Color frames either as png or jpg.
  0.jpg
  1.jpg
  ...
depth/          # 16 bit grayscale png images where values are in millimeters.
  0.png         # Depth frames might be smaller in size than the rgb frames.
  1.png
  ...
pose/
  0.txt         # 4 x 4 world to camera transform.
  1.txt
  ...
semantic/       # Ground truth semantic annotations provided by user.
  10.png        # These might not exist.
  150.png
intrinsics.txt  # 4 x 4 camera matrix.
bbox.txt        # 6 values denoting the bounds of the scene (min_x, min_y, min_z, max_x, max_y, max_z).
nerf/           # Contains NeRF checkpoints and training metadata.
```

## Computing camera poses

The script [`script/mapping.py`](script/mapping.py) defines a mapping pipeline which will compute camera poses for your scene. The required input files are:
- `rgb/` images
- `depth/` frames
- `intrinsics.txt` camera intrinsic parameters

The computed outputs are:
- `pose/` camera poses for each frame
- `bbox.txt` scene bounds


## Pretraining on a scene

To fit the representation to the scene without the user interface, you can run `scripts/train.py`. Checkpoints and metadata data will be stored in the scene folder under the `nerf` directory.

## Installing

The installation instructions were tested for Python 3.8 and 3.9.
Some dependencies are recommended to be installed through Anaconda. In your Anaconda env, you can install them with:
```
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

Install into your desired python environment with the following commands:
```
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
pushd Hierarchical-Localization/
python -m pip install -e .
popd

git submodule update --init --recursive
pushd torch_ngp
pip install -e .
bash scripts/install_ext.sh
popd
pip install -e .
```


