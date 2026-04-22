# TartanGround ANYmal Dataset Initialization

This guide documents how to initialize the TartanGround quadruped dataset pipeline for ANYmal in this repository.

It covers:
- Creating a conda environment named `tartan`
- Installing the local `tartanairpy` package (TartanAir library)
- Running `download.py`, `process_bag.py`, `process_imu.py`, and `process_img.py`
- Exact data paths required by each script

## 1) Create Conda Environment

Run from the repository root:

```bash
conda create -n tartan python=3.9 -y
conda activate tartan
```

## 2) Install TartanAir Library (Local)

The dataset repo is vendored in this workspace under `./tartanairpy`.

```bash
pip install -e ./tartanairpy
```

Additional packages used by processing scripts:

```bash
pip install rosbags pandas scipy opencv-python matplotlib
```

## 3) Run Pipeline

Run all commands from repository root:

```bash
python download.py
python process_bag.py
python process_imu.py
python process_img.py
python validate_processed_csvs.py
```

Optional diagnostics:

```bash
python test_bag.py
python test_imu.py
```

## 4) Data Path Requirements (Per Script)

### download.py

Purpose:
- Downloads ANYmal trajectories from TartanAir into local storage.

Key configuration in script:
- `PRIORITY_CANDIDATES`: environments to try.
- `TARGET_TRAJ`: list of trajectory IDs (empty list means all available).
- `TARGET_VERSION = ['anymal']`
- `TARGET_MODALITIES = ['rosbag', 'imu', 'meta']`
- `DOWNLOAD_ROOT = './anymal'`

<span style="color:red;"><strong>Caution:</strong></span> For a first run, try the pipeline with a single environment and trajectory (smaller download, easier to debug), for example `PRIORITY_CANDIDATES = ['ForestEnv']` and `TARGET_TRAJ = ['P2000']`.

Input requirements:
- Access to TartanAir API through `tartanair` initialization.

Output paths created:
- `anymal/<ENV>/Data_anymal/<TRAJ>/rosbags/*.bag`
- `anymal/<ENV>/Data_anymal/<TRAJ>/imu/*.npy`
- `anymal/<ENV>/Data_anymal/<TRAJ>/meta/*`

### process_bag.py

Purpose:
- Extracts `/state_estimator/anymal_state` from ROS bags to CSV.

Hardcoded path roots:
- Input root: `anymal`
- Output root: `processed`

Input path pattern required:
- `anymal/<ENV>/Data_anymal/<TRAJ>/rosbags/*.bag`

Output path pattern produced:
- `processed/<ENV>/<TRAJ>/<BAG_STEM>_bag.csv`

Notes:
- Script skips trajectories with missing `rosbags/` folders.
- Topic must exist in bag: `/state_estimator/anymal_state`.

### process_imu.py

Purpose:
- Converts IMU arrays to FLU convention and writes `imu.csv`.

Hardcoded path roots:
- Input root: `anymal`
- Output root: `processed`

Input path pattern required:
- `anymal/<ENV>/Data_anymal/<TRAJ>/imu/`

Required files under each IMU directory:
- `imu_time.npy`
- `acc.npy`
- `acc_nograv.npy`
- `gyro.npy`
- `pos_global.npy`
- `vel_global.npy`
- `ori_global.npy`

Output path pattern produced:
- `processed/<ENV>/<TRAJ>/imu.csv`

Timestamp schema written:
- Columns are `sec` and `nanosec`.
- Reconstructed time is `sec + nanosec * 1e-9`.

### process_img.py

Purpose:
- Moves and renames camera frames; optionally creates synchronized animation.

Hardcoded path roots:
- Input root: `ForestEnv_Camera`
- Output root: `processed`

Input path pattern required for images:
- `ForestEnv_Camera/<ENV>/Data_anymal/<TRAJ>/image_lcam_front/*.png`

Input timestamp dependency required:
- `ForestEnv_Camera/<ENV>/Data_anymal/<TRAJ>/imu/cam_time.npy`

Output paths produced:
- `processed/<ENV>/<TRAJ>/frames/*.png`
- Optional animation file: `processed/<ENV>/<TRAJ>/animation_<SPEED>x.mp4`

Important:
- `GENERATE_ANIMATION = False` by default, so only frame migration/renaming runs unless changed in script.
- For animation mode, script also expects:
  - `processed/<ENV>/<TRAJ>/imu.csv`
  - `processed/<ENV>/<TRAJ>/*_bag.csv`

## 5) Expected Directory Layout

After download and processing, the expected structure is:

```text
anymal/
  <ENV>/
    Data_anymal/
      <TRAJ>/
        rosbags/*.bag
        imu/*.npy
        meta/*

ForestEnv_Camera/
  <ENV>/
    Data_anymal/
      <TRAJ>/
        image_lcam_front/*.png
        imu/cam_time.npy

processed/
  <ENV>/
    <TRAJ>/
      imu.csv
      *_bag.csv
      frames/*.png
      animation_<SPEED>x.mp4   # only if enabled
```

## 6) Validation and Troubleshooting

Validation command:

```bash
python validate_processed_csvs.py --processed-root processed
```

Common issues:
- `Root directory not found`: run from repository root or create expected folder (`anymal` or `ForestEnv_Camera`).
- Missing `rosbags/*.bag`: re-run `download.py` and check chosen environments/trajectories.
- Missing IMU `.npy` files: ensure `TARGET_MODALITIES` includes `imu` in `download.py`.
- Image/timestamp mismatch in `process_img.py`: number of `*.png` files must match entries in `cam_time.npy`.

## 7) Minimal Quick Start

```bash
conda create -n tartan python=3.9 -y
conda activate tartan
pip install -e ./tartanairpy
pip install rosbags pandas scipy opencv-python matplotlib
# modify download script for desired modalities
python download.py
python process_bag.py
python process_imu.py
python process_img.py
python validate_processed_csvs.py
```
