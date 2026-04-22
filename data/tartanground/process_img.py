import cv2
import numpy as np
import pandas as pd
import re
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ==========================================
# CONFIGURATION
# ==========================================
SPEED_MULTIPLIER = 2  # Control playback speed (Integer values e.g., 1 for 1x, 2 for 2x, 3 for 3x)
GENERATE_ANIMATION = False  # If False, only migrate/rename frames and skip MP4 rendering

root_path = Path("anymal")
output_root = Path("processed")

if not root_path.exists():
    print(f"[FATAL] Root directory not found: {root_path}")
    exit(1)

def extract_num(filepath):
    match = re.search(r'\d+', filepath.name)
    return int(match.group()) if match else 0


def to_safe_timestamp_filename(timestamp_sec):
    # Keep float precision while producing a filename-safe integer/fraction format.
    if abs(timestamp_sec) < 1e-12:
        timestamp_sec = 0.0

    ts = format(float(timestamp_sec), ".17f").rstrip("0").rstrip(".")
    if not ts:
        ts = "0"

    if "." in ts:
        sec_part, frac_part = ts.split(".", 1)
    else:
        sec_part, frac_part = ts, "0"

    sec_digits = "".join(ch for ch in sec_part if ch.isdigit()) or "0"
    frac_digits = "".join(ch for ch in frac_part if ch.isdigit()) or "0"

    return f"{sec_digits}_{frac_digits}.png"

for traj_dir in root_path.glob("*/Data_anymal/*"):
    if not traj_dir.is_dir():
        continue
        
    env_name = traj_dir.parents[1].name
    traj_name = traj_dir.name
    out_dir = output_root / env_name / traj_name

    cam_src_dir = traj_dir / "image_lcam_front"
    frames_dir = out_dir / "frames"
    cam_time_path = traj_dir / "imu" / "cam_time.npy"

    if not all(p.exists() for p in [cam_src_dir, cam_time_path]):
        print(f"[WARNING] Missing required source files for {traj_name}. Skipping.")
        continue

    # Load camera timestamps and source image paths.
    cam_t = np.load(cam_time_path).flatten()
    img_paths = sorted(cam_src_dir.glob("*.png"), key=extract_num)
    if len(img_paths) != len(cam_t):
        print(f"[INVALID] Camera image count ({len(img_paths)}) != timestamp count ({len(cam_t)}). Skipping.")
        continue

    # Always use zero-normalized camera timestamps for destination frame names.
    cam_t_norm = cam_t - cam_t[0] if len(cam_t) > 0 else cam_t

    # Replace destination frames directory if it already exists.
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    moved_img_paths = []
    used_names = {}
    for src_img, t_norm in zip(img_paths, cam_t_norm):
        dst_name = to_safe_timestamp_filename(t_norm)
        if dst_name in used_names:
            used_names[dst_name] += 1
            stem = Path(dst_name).stem
            dst_name = f"{stem}_dup{used_names[dst_name]}.png"
        else:
            used_names[dst_name] = 0

        dst_path = frames_dir / dst_name
        shutil.move(str(src_img), str(dst_path))
        moved_img_paths.append(dst_path)

    print(f"[ACTION] -> Moved {len(moved_img_paths)} frames to {frames_dir}")

    if not GENERATE_ANIMATION:
        print(f"[ACTION] -> Animation disabled for {traj_name}. Skipping MP4 generation.")
        continue

    imu_csv_path = out_dir / "imu.csv"

    # Locate bag CSV
    bag_csvs = list(out_dir.glob("*_bag.csv"))
    if not bag_csvs:
        print(f"[WARNING] No bag CSV found in {out_dir}. Skipping animation.")
        continue
    bag_csv_path = bag_csvs[0]

    if not all(p.exists() for p in [imu_csv_path, bag_csv_path]):
        print(f"[WARNING] Missing IMU or bag CSV for {traj_name}. Skipping animation.")
        continue

    imu_df = pd.read_csv(imu_csv_path)
    imu_t = imu_df['sec'].values + imu_df['nanosec'].values * 1e-9
    imu_x, imu_y = imu_df['pos_x'].values, imu_df['pos_y'].values

    bag_df = pd.read_csv(bag_csv_path)
    bag_t = bag_df['sec'].values + bag_df['nanosec'].values * 1e-9
    bag_x, bag_y = bag_df['p_x'].values, bag_df['p_y'].values

    # Temporal Epoch Alignment Check
    if len(cam_t) > 0 and len(bag_t) > 0:
        if abs(cam_t[0] - bag_t[0]) > 1000:
            print(f"[WARNING] Epoch mismatch in {traj_name}. Normalizing all times to relative 0.")
            cam_t = cam_t - cam_t[0]
            imu_t = imu_t - imu_t[0]

    # Calculate base FPS BEFORE decimation to preserve physical time mapping
    mean_dt = np.mean(np.diff(cam_t))
    fps = int(round(1.0 / mean_dt)) if mean_dt > 0 else 30

    # Decimate arrays for desired speed multiplier
    cam_t = cam_t[::SPEED_MULTIPLIER]
    img_paths = moved_img_paths[::SPEED_MULTIPLIER]

    # Setup 1x3 Matplotlib Figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Plot 1: Bag GT
    ax1.plot(bag_x, bag_y, color='gray', alpha=0.5, label='Full Trajectory')
    pt_bag, = ax1.plot([], [], 'ro', markersize=8, label='Current Pos')
    ax1.set_title("Bag Groundtruth (p_x, p_y)")
    ax1.set_xlabel("p_x"); ax1.set_ylabel("p_y")
    ax1.axis('equal'); ax1.legend()

    # Plot 2: IMU GT
    ax2.plot(imu_x, imu_y, color='gray', alpha=0.5, label='Full Trajectory')
    pt_imu, = ax2.plot([], [], 'bo', markersize=8, label='Current Pos')
    ax2.set_title("IMU Global Pos (pos_x, pos_y)")
    ax2.set_xlabel("pos_x"); ax2.set_ylabel("pos_y")
    ax2.axis('equal'); ax2.legend()

    # Plot 3: Camera Front
    ax3.axis('off')
    ax3.set_title("Camera Front")
    img_plot = None

    plt.tight_layout()

    # Video Writer Initialization
    output_file = out_dir / f"animation_{SPEED_MULTIPLIER}x.mp4"
    width, height = fig.canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    print(f"[ACTION] -> Rendering {len(cam_t)} frames for {traj_name} at {SPEED_MULTIPLIER}x speed...")

    # Render Loop
    for t_curr, img_p in zip(cam_t, img_paths):
        # Match nearest timestamps
        idx_bag = np.argmin(np.abs(bag_t - t_curr))
        idx_imu = np.argmin(np.abs(imu_t - t_curr))

        pt_bag.set_data([bag_x[idx_bag]], [bag_y[idx_bag]])
        pt_imu.set_data([imu_x[idx_imu]], [imu_y[idx_imu]])

        # Update Image
        img = cv2.imread(str(img_p))
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img_plot is None:
            img_plot = ax3.imshow(img_rgb)
        else:
            img_plot.set_data(img_rgb)

        # Draw to canvas and push to video
        canvas.draw()
        frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape((height, width, 4))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        video.write(frame_bgr)

    video.release()
    plt.close(fig)
    print(f"[ACTION] -> Saved synchronized plot to {output_file}")