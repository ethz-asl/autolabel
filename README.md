# Autolabel

The goal of this project is to provide accurate ground truth data for RGB-D sensor streams.

## User interface

The main way to interact with this project is through the graphical user interface.

Run the user interface with `python scripts/gui.py <scene>`

## Scene directory structure

The scene directory structure is as follows:
```
rgb/            # Undistorted color frames either as png or jpg.
  00000.jpg
  00001.jpg
  ...
raw_rgb/        # Raw distorted color frames.
depth/          # 16 bit grayscale png images where values are in millimeters.
  00000.png     # Depth frames might be smaller in size than the rgb frames.
  00001.png     # Undistorted to match a perfect pinhole camera model.
  ...
raw_depth/      # Raw original distorted depth frames.
pose/
  00000.txt         # 4 x 4 world to camera transform.
  00001.txt
  ...
semantic/       # Ground truth semantic annotations provided by user.
  00010.png        # These might not exist.
  00150.png
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

