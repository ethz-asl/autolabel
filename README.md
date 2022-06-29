# Autolabel

The goal of this project is to provide accurate ground truth data for RGB-D sensor streams.

## User Interface

The main way to interact with this project is through the graphical user interface.

Run the user interface with `python scripts/gui.py <scene>`

## Scene Directory Structure

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
semantic/       # Ground truth sparse semantic annotations provided by user.
  10.png        # These might not exist.
  150.png
gt_masks/       # Optional
  10.json       # Dense ground truth masks used for evaluation.
  150.json      # Used e.g. by scripts/evaluate.py
intrinsics.txt  # 4 x 4 camera matrix.
bbox.txt        # 6 values denoting the bounds of the scene (min_x, min_y, min_z, max_x, max_y, max_z).
nerf/           # Contains NeRF checkpoints and training metadata.
```

## Pretraining on a Scene

To fit the representation to the scene without the user interface, you can run `scripts/train.py`. Checkpoints and metadata data will be stored in the scene folder under the `nerf` directory.

## Installing

Some dependencies are recommended to be installed through Anaconda. In your Anaconda env, you can install them with:
```
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

Install into your desired python environment with the following commands:
```
git submodule update --init --recursive
pushd torch_ngp
pip install -e .
bash scripts/install_ext.sh
popd
pip install -e .
```

## Evaluating against ground truth frames

We use [labelme](https://github.com/wkentaro/labelme) to annotate ground truth frames. To annotate frames, run:
```
labelme rgb --nodata --autosave --output gt_masks
```
inside a scene directory, to annotate the frames in the `rgb` folder. Corresponding annotations will be saved into the `gt_masks` folder. You don't need to annotate every single frame, but can sample just a few.

To compute the intersection-over-union agreement against the manually annotated frames, run:
```
python scripts/evaluate.pt <scene1> <scene2> # ...
```


