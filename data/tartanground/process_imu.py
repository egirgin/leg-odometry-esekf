import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R

root_path = Path("anymal")
output_root = Path("processed")

if not root_path.exists():
    print(f"[FATAL] Root directory not found: {root_path}")
    exit(1)

def load_npy(target_dir, filename):
    filepath = target_dir / filename
    return np.load(filepath) if filepath.exists() else None


def split_time_to_sec_nanosec(time_values):
    sec = np.floor(time_values).astype(np.int64)
    frac = time_values - sec.astype(np.float64)
    nanosec = np.round(frac * 1e9).astype(np.int64)

    # Handle rounding boundary (e.g., 0.9999999996 -> 1_000_000_000 ns).
    carry = nanosec // 1_000_000_000
    sec = sec + carry
    nanosec = nanosec % 1_000_000_000

    return sec, nanosec

# Transformation Matrix T: Right-Handed FRD to Right-Handed FLU
# 180-degree rotation around X-axis. det(T) = 1
T = np.array([
    [ 1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0, -1]
])

for traj_dir in root_path.glob("*/Data_anymal/*"):
    if not traj_dir.is_dir():
        continue
        
    imu_dir = traj_dir / "imu"
    if not imu_dir.exists():
        continue

    env_name = traj_dir.parents[1].name
    traj_name = traj_dir.name

    out_dir = output_root / env_name / traj_name
    out_dir.mkdir(parents=True, exist_ok=True)

    time = load_npy(imu_dir, "imu_time.npy")
    if time is None:
        continue
    time = time.flatten()
    
    acc = load_npy(imu_dir, "acc.npy")
    acc_nograv = load_npy(imu_dir, "acc_nograv.npy")
    gyro = load_npy(imu_dir, "gyro.npy")
    pos_global = load_npy(imu_dir, "pos_global.npy")
    vel_global = load_npy(imu_dir, "vel_global.npy")
    ori_global = load_npy(imu_dir, "ori_global.npy")

    if any(v is None for v in [acc, acc_nograv, gyro, pos_global, vel_global, ori_global]):
        print(f"[WARNING] Incomplete numpy arrays in {imu_dir}. Skipping.")
        continue

    n_samples = len(time)
    arrays = {"acc": acc, "acc_nograv": acc_nograv, "gyro": gyro, "pos_global": pos_global, "vel_global": vel_global, "ori_global": ori_global}

    if any(len(arr) != n_samples for arr in arrays.values()):
        print(f"[INVALID] Alignment assumption failed in {imu_dir}. Skipping.")
        continue

    # Zero position offset
    pos_global = pos_global - pos_global[0]

    # ==========================================
    # ORIENTATION TRANSFORMATION (FRD Euler -> FLU Quaternion)
    # ==========================================
    # Assuming ori_global is ZYX (Yaw-Pitch-Roll) based on prior analysis
    yaw_frd   = ori_global[:, 2]
    pitch_frd = ori_global[:, 1]
    roll_frd  = ori_global[:, 0]
    
    # Create rotation matrices in the original FRD frame
    r_frd = R.from_euler('ZYX', np.column_stack([yaw_frd, pitch_frd, roll_frd]), degrees=False)
    
    # Transform basis to FLU: R_flu = T * R_frd * T^T
    R_flu_mats = np.einsum('ij,njk,kl->nil', T, r_frd.as_matrix(), T.T)
    quats_flu = R.from_matrix(R_flu_mats).as_quat()

    sec, nanosec = split_time_to_sec_nanosec(time)

    # ==========================================
    # APPLY FRD TO FLU TRANSFORMATION (X -> X, Y -> -Y, Z -> -Z)
    # ==========================================
    df = pd.DataFrame({
        'sec': sec,
        'nanosec': nanosec,
        
        # LINEAR VECTORS (Pos, Vel, Accel)
        'accel_x': acc[:, 0], 'accel_y': -acc[:, 1], 'accel_z': -acc[:, 2],
        'accel_nograv_x': acc_nograv[:, 0], 'accel_nograv_y': -acc_nograv[:, 1], 'accel_nograv_z': -acc_nograv[:, 2],
        'pos_x': pos_global[:, 0], 'pos_y': -pos_global[:, 1], 'pos_z': -pos_global[:, 2],
        'vel_x': vel_global[:, 0], 'vel_y': -vel_global[:, 1], 'vel_z': -vel_global[:, 2],
        
        # PSEUDOVECTORS (Gyro) - Identical mapping due to proper rotation
        'gyro_x': gyro[:, 0], 'gyro_y': -gyro[:, 1], 'gyro_z': -gyro[:, 2],
        
        # ORIENTATION (Quaternion for EKF)
        'ori_qx': quats_flu[:, 0],
        'ori_qy': quats_flu[:, 1],
        'ori_qz': quats_flu[:, 2],
        'ori_qw': quats_flu[:, 3]
    })

    output_file = out_dir / "imu.csv"
    df.to_csv(output_file, index=False)
    print(f"[ACTION] -> Mapped {imu_dir} successfully to Right-Handed FLU at {output_file}")