# Autolabel

The goal of this project is to provide accurate ground truth data for RGB-D sensor streams.

https://user-images.githubusercontent.com/1204635/191912816-0de3791c-d29b-458a-aead-ba020a0cc871.mp4

## Getting started

### Installing

The installation instructions were tested for Python 3.8 and 3.9.
Some dependencies are recommended to be installed through Anaconda. In your Anaconda env, you can install them with:
```
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

Install into your desired python environment with the following commands:
```
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
git clone --recursive https://github.com/francescomilano172/Hierarchical-Localization
pushd Hierarchical-Localization/
git checkout feature/arbitrary_camera_model
python -m pip install -e .
popd

git submodule update --init --recursive
pushd torch_ngp
pip install -e .
bash scripts/install_ext.sh
popd

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

pip install -e .
```

### Usage

After installing the project using the instructions above, you can follow these steps to run autolabel on an example scene.

```
# Download example scene
wget http://robotics.ethz.ch/~asl-datasets/2022_autolabel/bench.tar.gz
# Uncompress
tar -xvf bench.tar.gz

# Compute camera poses, scene bounds and undistort images using raw input images
python scripts/mapping.py bench

# Compute DINO features from color images.
python scripts/compute_feature_maps.py bench --features dino --autoencode
# Pretrain neural representation on color, depth and extracted features
python scripts/train.py bench --features dino

# Open the scene in the graphical user interface for annotation
python scripts/gui.py bench --features dino
```

### Keybindings

The GUI can be controlled with the following keybindings:

| Key          | Class Name                    |
| ------------ | ----------------------------- |
| `0`          | select background paint brush |
| `1`          | select foreground paint brush |
| `esc` or `Q` | shutdown application          |
| `ctrl+S`     | save model                    |
| `C`          | clear image                   |


## Scene directory structure

The scene directory structure is as follows:
```
raw_rgb/        # Raw distorted color frames.
rgb/            # Undistorted color frames either as png or jpg.
  00000.jpg
  00001.jpg
  ...
raw_depth/      # Raw original distorted depth frames.
  00000.png     # 16 bit grayscale png images where values are in millimeters.
  00001.png     # Depth frames might be smaller in size than the rgb frames.
  ...
depth/          # Undistorted frames to match a perfect pinhole camera model.
  00000.png
  00001.png
  ...
pose/
  00000.txt       # 4 x 4 world to camera transform.
  00001.txt
  ...
semantic/         # Ground truth semantic annotations provided by user.
  00010.png       # These might not exist.
  00150.png
gt_masks/         # Optional
  00010.json      # Dense ground truth masks used for evaluation.
  00150.json      # Used e.g. by scripts/evaluate.py
intrinsics.txt    # 4 x 4 camera matrix.
bbox.txt          # 6 values denoting the bounds of the scene (min_x, min_y, min_z, max_x, max_y, max_z).
nerf/             # Contains NeRF checkpoints and training metadata.
```

## Computing camera poses

The script [`scripts/mapping.py`](scripts/mapping.py) defines a mapping pipeline which will compute camera poses for your scene. The required input files are:
- `raw_rgb/` images
- `raw_depth/` frames
- `intrinsics.txt` camera intrinsic parameters

The computed outputs are:
- `rgb/` undistorted camera images
- `depth/` undistorted depth images
- `pose/` camera poses for each frame
- `intrinsics.txt` inferred camera intrinsic parameters
- `bbox.txt` scene bounds

### Recording data

If you have a LiDAR enabled iOS device, you can use the [Stray Scanner](https://apps.apple.com/us/app/stray-scanner/id1557051662) app to record data. The script at `scripts/convert_scanner.py` will allow you to convert a scene recorded using the app to the above format. You can then run the `mapping.py` script to run structure from motion and compute the other outputs.

## Pretraining on a scene

To fit the representation to the scene without the user interface, you can run `scripts/train.py`. Checkpoints and metadata data will be stored in the scene folder under the `nerf` directory.


To use pretrained features as additional training supervision, pretrain on these and then open the scene in the GUI, run:
```
python scripts/compute_feature_maps.py --features dino --autoencode <scene>
python scripts/train.py --features dino <scene>
python scripts/gui.py --features dino <scene>
```

The models are saved in the scene folder under the `nerf` directory, organized according to the given parameters. I.e. the gui will load the model which matches the given parameters. If one is not found, it will simply randomly initialize the network.


## Evaluating against ground truth frames

We use [labelme](https://github.com/wkentaro/labelme) to annotate ground truth frames. Follow the installation instructions, using for instance a `conda` environment, and making sure that your Python version is `<3.10` to avoid type errors (see [here](https://github.com/wkentaro/labelme/issues/1020#issuecomment-1139749978)). To annotate frames, run:
```
labelme rgb --nodata --autosave --output gt_masks
```
inside a scene directory, to annotate the frames in the `rgb` folder. Corresponding annotations will be saved into the `gt_masks` folder. You don't need to annotate every single frame, but can sample just a few.

To compute the intersection-over-union agreement against the manually annotated frames, run:
```
python scripts/evaluate.py <scene1> <scene2> # ...
```

## Code formatting

This repository enforces code formatting rules using [`yapf`](https://github.com/google/yapf). After installing, you can format the code before committing by running:
```
yapf --recursive autolabel scripts -i
```

### Vim

In case you want to automatically run formatting in Vim on save, you can follow these steps.

First, install `google/yapf` as a vim plugin. If using Vundle, add `Plugin 'google/yapf'` to your `.vimrc` and run `:PluginInstall`.

Copy the file `.yapf.vim` to `$HOME/.vim/autoload/yapf.vim`, creating the autoload directory if it doesn't exist.

To run yapf on save for Python files, add `autocmd FileType python autocmd BufWritePre <buffer> call yapf#YAPF()` to your `.vimrc` then restart vim.

## Research Papers

Baking in the Feature: Accelerating Volumetric Segmentation by Rendering Feature Maps - [Link](https://keke.dev/baking-in-the-feature)
