
# Importing data

## Capturing your own datasets

If you have a LiDAR enabled iOS device, you can use the [Stray Scanner](https://apps.apple.com/us/app/stray-scanner/id1557051662) app to record data. The script at `scripts/convert_scanner.py` will allow you to convert a scene recorded using the app to the above format. You can then run the `mapping.py` script to run structure from motion and compute the other outputs.

After capturing and moving the scenes over to your computer, convert to Autolabel format with:
```
python scripts/convert_scanner.py <scanner-scene> --out scenes/scene_name/

# Compute camera poses.
python scripts/mapping.py scenes/scene_name/
```

## Importing ARKitScenes scenes

Here are the steps required to download and import scenes from the ARKitScenes dataset.

```
# Clone ARKitScenes repository
git clone https://github.com/apple/ARKitScenes.git arkit-scenes
cd arkit-scenes

# Create a directory for the scenes to download them
mkdir -p scenes/raw/ && mkdir -p scenes/converted

# Download the required parts of the dataset
# For now, we only download the low resolution RGB images (256x192), but higher
# resolution frames could be used.
python download_data.py raw --split Training \
  --video_id_csv depth_upsampling/upsampling_train_val_splits.csv \
  --download_dir scenes/raw \
  --raw_dataset_assets lowres_wide lowres_depth lowres_wide.traj confidence lowres_wide_intrinsics

# Convert the ARKitScenes to the format used by Autolabel
python scripts/convert_arkitscenes.py scenes/raw/ --out scenes/converted/
```

## Importing Replica renders

We have written data conversion scripts for different publicly available datasets.

Renders from the [Replica](https://github.com/facebookresearch/Replica-Dataset) dataset published by [SemanticNeRF](https://github.com/Harry-Zhi/semantic_nerf/) can be converted using the `scripts/convert_replica.py` script.

## Importing ScanNet scenes

ScanNet scenes can be imported with the `scripts/convert_scannet.py` script.

It is used as:
```
python scripts/convert_scannet.py <scannet-directory> --label-map <path-to-labelmap.tsv> --out <path-to-output-directory> --stride <N>
```

- `scannet-directory` this is the path to the raw scannet dataset which contains each scene as a subdirectory. Each scene in turn contains the `*-label-filt.zip`, `*.sens` etc. files in the ScanNet format.
- `--out` specifies the output directory. Each scene will be stored as a subdirectory.
- `--stride` is an integer parameter specifying how many frames to keep. Only every Nth frame is kept in the scan.

This will in addition also create `mesh.ply` and `mesh_labels.npy` files use by the vision-language evaluation scripts.

