import os
import glob
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def skew_symmetric(v):
    """Creates a 3x3 skew-symmetric matrix from a 3-element vector."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])

# -----------------------------------------------------------------------------
# 1. SETUP & DATA LOADING
# -----------------------------------------------------------------------------
# Hardcoded to your specific WSL path
BASE_DIR = "/mnt/c/Users/91935/OneDrive/Desktop/ML_Dead_Reckoning"
EUROC_DIR = os.path.join(BASE_DIR, "EuRoC MAV dataset")

# Separate file discovery for IMU and Ground Truth
imu_files = glob.glob(os.path.join(EUROC_DIR, "**", "imu0", "*.csv"), recursive=True) + \
            glob.glob(os.path.join(EUROC_DIR, "**", "*imu*.csv"), recursive=True)

gt_files = glob.glob(os.path.join(EUROC_DIR, "**", "*gt*.csv"), recursive=True) + \
           glob.glob(os.path.join(EUROC_DIR, "**", "*groundtruth*.csv"), recursive=True)

imu_path = imu_files[0] if imu_files else os.path.join(EUROC_DIR, "data.csv")
gt_path = gt_files[0] if gt_files else None

print(f"Loading IMU data from          : {imu_path}")
df_imu = pd.read_csv(imu_path).apply(pd.to_numeric, errors='coerce').dropna()

# Extract IMU columns (Timestamp, Gyro X,Y,Z, Accel X,Y,Z)
timestamps = df_imu.iloc[:, 0].values
gyro_data = df_imu.iloc[:, 1:4].values
accel_data = df_imu.iloc[:, 4:7].values

# Extract GT columns
if gt_path and os.path.exists(gt_path):
    print(f"Loading Ground Truth data from : {gt_path}")
    df_gt = pd.read_csv(gt_path).apply(pd.to_numeric, errors='coerce').dropna()
    gt_pos = df_gt.iloc[:, 1:4].values  # Position p_x, p_y, p_z
    gt_quat = df_gt.iloc[:, 4:8].values # Quaternions q_w, q_x, q_y, q_z
    
    # EuRoC Ground truth velocities are typically in columns 8, 9, 10
    if df_gt.shape[1] >= 11:
        gt_vel = df_gt.iloc[:, 8:11].values 
    else:
        gt_vel = None
else:
    gt_pos = None
    gt_quat = None
    gt_vel = None

num_frames = len(timestamps)

# -----------------------------------------------------------------------------
# 2. STATE INITIALIZATION (GROUND TRUTH ALIGNED)
# -----------------------------------------------------------------------------
# Initialize nominal states
p = gt_pos[0].copy() if gt_pos is not None else np.zeros(3)
v = gt_vel[0].copy() if gt_vel is not None else np.zeros(3)  # Initial velocity

# Initialize orientation from ground truth if available, else identity
if gt_quat is not None:
    # Convert from EuRoC [w, x, y, z] to scipy [x, y, z, w] format
    q_init = gt_quat[0]
    if len(q_init) == 4:
        q = R.from_quat([q_init[1], q_init[2], q_init[3], q_init[0]])
    else:
        q = R.from_quat([0, 0, 0, 1])
else:
    q = R.from_quat([0, 0, 0, 1])

bg = np.zeros(3)                 # Gyro bias
ba = np.zeros(3)                 # Accel bias
g = np.array([0.0, 0.0, -9.81])  # Gravity vector (m/s^2)

# Error State Covariance Matrix (15x15)
P = np.eye(15) * 0.01

# Process Noise Covariance (Q) - ADIS16448 IMU specs
noise_acc = 0.04
noise_gyr = 0.004
noise_ba = 0.002
noise_bg = 0.0004
Q = np.diag([noise_acc**2]*3 + [noise_gyr**2]*3 + [noise_ba**2]*3 + [noise_bg**2]*3)

# Trajectory & Velocity storage
pred_trajectory = np.zeros((num_frames, 3))
pred_trajectory[0] = p

pred_velocity = np.zeros((num_frames, 3))
pred_velocity[0] = v

# -----------------------------------------------------------------------------
# 3. ESKF PREDICTION (DEAD RECKONING LOOP)
# -----------------------------------------------------------------------------
print(f"Running ESKF over {num_frames} frames...")

for i in range(num_frames - 1):
    # Dynamic timestamp step calculation (Converts nanoseconds to seconds if needed)
    raw_dt = timestamps[i+1] - timestamps[i]
    dt = raw_dt / 1e9 if raw_dt > 1e3 else raw_dt
    
    # Guard against invalid or duplicate timestamps
    if dt <= 0 or dt > 0.1:
        dt = 0.005  # Fallback to 200 Hz default
    
    # Unbiased IMU measurements
    acc = accel_data[i] - ba
    omega = gyro_data[i] - bg
    
    # Rotation matrix from current orientation
    C = q.as_matrix()
    
    # 1. Update Nominal State (Kinematics Propagation)
    acc_world = C @ acc + g
    p = p + v * dt + 0.5 * acc_world * (dt**2)
    v = v + acc_world * dt
    
    # Update Orientation
    delta_q = R.from_rotvec(omega * dt)
    q = q * delta_q
    
    # 2. Construct Error-State Jacobian (F)
    F = np.eye(15)
    F[0:3, 3:6] = np.eye(3) * dt
    F[3:6, 6:9] = -C @ skew_symmetric(acc) * dt
    F[3:6, 9:12] = -C * dt
    F[6:9, 6:9] = R.from_rotvec(-omega * dt).as_matrix()
    F[6:9, 12:15] = -np.eye(3) * dt
    
    # Noise Jacobian (L)
    L = np.zeros((15, 12))
    L[3:6, 0:3] = C * dt
    L[6:9, 3:6] = np.eye(3) * dt
    L[9:12, 6:9] = np.eye(3) * dt
    L[12:15, 9:12] = np.eye(3) * dt
    
    # 3. Propagate Covariance
    P = F @ P @ F.T + L @ Q @ L.T
    
    # Store trajectory and velocity frames
    pred_trajectory[i+1] = p
    pred_velocity[i+1] = v

# -----------------------------------------------------------------------------
# 4. EXPORT & EVALUATE
# -----------------------------------------------------------------------------
output_path = os.path.join(BASE_DIR, "eskf_euroc_trajectory.csv")
pd.DataFrame(pred_trajectory, columns=["x", "y", "z"]).to_csv(output_path, index=False)
print(f"Exported predicted path to: {output_path}")

# Run metrics evaluation
if gt_pos is not None:
    min_len = min(len(pred_trajectory), len(gt_pos))
    pos_err = np.linalg.norm(pred_trajectory[:min_len] - gt_pos[:min_len], axis=1)
    
    # Root Mean Square Error (RMSE) & Absolute Trajectory Error (ATE) for Position
    rmse_pos = np.sqrt(np.mean(pos_err**2))
    ate = rmse_pos  
    max_err = np.max(pos_err)
    
    # Calculate Velocity RMSE if available
    if gt_vel is not None:
        min_len_v = min(len(pred_velocity), len(gt_vel))
        vel_err = np.linalg.norm(pred_velocity[:min_len_v] - gt_vel[:min_len_v], axis=1)
        rmse_vel = np.sqrt(np.mean(vel_err**2))
        vel_str = f"{rmse_vel:.4f} m/s"
    else:
        vel_str = "N/A (Ground Truth Missing)"
    
    # Calculate Total Distance Traveled
    step_lens = np.linalg.norm(np.diff(gt_pos[:min_len], axis=0), axis=1)
    total_dist = np.sum(step_lens)
    drift_pct = (pos_err[-1] / total_dist) * 100 if total_dist > 0 else 0.0
    
    print("=" * 55)
    print("      ESKF DEAD RECKONING EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Total Trajectory Length : {total_dist:.2f} m")
    print(f"  Position RMSE (m)       : {rmse_pos:.4f} m")
    print(f"  Velocity RMSE (m/s)     : {vel_str}")
    print(f"  ATE (m)                 : {ate:.4f} m")
    print(f"  Max Pos Error (m)       : {max_err:.4f} m")
    print(f"  Final Drift (%)         : {drift_pct:.2f} %")
    print("=" * 55)
else:
    print("Ground truth position columns not found for evaluation.")