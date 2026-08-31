import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def skew_symmetric(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])

# -----------------------------------------------------------------------------
# 1. SETUP & DATA LOADING
# -----------------------------------------------------------------------------
BASE_DIR = "/mnt/c/Users/91935/OneDrive/Desktop/ML_Dead_Reckoning"
DATA_DIR = os.path.join(BASE_DIR, "Dataset 1 (Asending Square Pattern)")

imu_path = os.path.join(DATA_DIR, "rs_imu.csv")
gt_path = os.path.join(DATA_DIR, "mocap_vehicle_data.csv")

print(f"Loading IMU data from          : {imu_path}")
df_imu = pd.read_csv(imu_path).apply(pd.to_numeric, errors='coerce').dropna()

# Extract IMU: t, a_x, a_y, a_z, w_x, w_y, w_z
timestamps = df_imu.iloc[:, 0].values
accel_data = df_imu.iloc[:, 1:4].values
gyro_data = df_imu.iloc[:, 4:7].values

print(f"Loading Ground Truth data from : {gt_path}")
df_gt = pd.read_csv(gt_path).apply(pd.to_numeric, errors='coerce').dropna()

# Extract GT: t, p_x, p_y, p_z, q_w, q_x, q_y, q_z
gt_timestamps = df_gt.iloc[:, 0].values
gt_pos = df_gt.iloc[:, 1:4].values  
gt_quat = df_gt.iloc[:, 4:8].values 

# Compute GT Velocity via finite difference (since it is missing in the dataset)
gt_vel = np.zeros_like(gt_pos)
for j in range(1, len(gt_pos)):
    dt_gt = gt_timestamps[j] - gt_timestamps[j-1]
    dt_gt_sec = dt_gt / 1e9 if dt_gt > 1e3 else dt_gt
    if dt_gt_sec > 0:
        gt_vel[j] = (gt_pos[j] - gt_pos[j-1]) / dt_gt_sec
gt_vel[0] = gt_vel[1]  # Initial velocity fallback

num_frames = len(timestamps)

# -----------------------------------------------------------------------------
# 2. STATE INITIALIZATION (GROUND TRUTH ALIGNED)
# -----------------------------------------------------------------------------
p = gt_pos[0].copy() 
v = gt_vel[0].copy()

# GT quaternion format in dataset is [w, x, y, z]
q_init = gt_quat[0]
q = R.from_quat([q_init[1], q_init[2], q_init[3], q_init[0]])

bg = np.zeros(3)                 
ba = np.zeros(3)                 
g = np.array([0.0, 0.0, -9.81])  

P = np.eye(15) * 0.01

noise_acc = 0.04
noise_gyr = 0.004
noise_ba = 0.002
noise_bg = 0.0004
Q = np.diag([noise_acc**2]*3 + [noise_gyr**2]*3 + [noise_ba**2]*3 + [noise_bg**2]*3)

pred_trajectory = np.zeros((num_frames, 3))
pred_velocity = np.zeros((num_frames, 3))

pred_trajectory[0] = p
pred_velocity[0] = v

# -----------------------------------------------------------------------------
# 3. ESKF PREDICTION (DEAD RECKONING LOOP)
# -----------------------------------------------------------------------------
print(f"Running ESKF over {num_frames} frames...")

for i in range(num_frames - 1):
    raw_dt = timestamps[i+1] - timestamps[i]
    dt = raw_dt / 1e9 if raw_dt > 1e3 else raw_dt
    if dt <= 0 or dt > 0.1:
        dt = 0.005  
    
    acc = accel_data[i] - ba
    omega = gyro_data[i] - bg
    
    C = q.as_matrix()
    
    # 1. Update Nominal State
    acc_world = C @ acc + g
    p = p + v * dt + 0.5 * acc_world * (dt**2)
    v = v + acc_world * dt
    
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
    
    pred_trajectory[i+1] = p
    pred_velocity[i+1] = v

# -----------------------------------------------------------------------------
# 4. EVALUATION
# -----------------------------------------------------------------------------
min_len = min(len(pred_trajectory), len(gt_pos))

pos_err = np.linalg.norm(pred_trajectory[:min_len] - gt_pos[:min_len], axis=1)
rmse_pos = np.sqrt(np.mean(pos_err**2))
ate = rmse_pos  
final_drift_dist = pos_err[-1]

vel_err_x = pred_velocity[:min_len, 0] - gt_vel[:min_len, 0]
vel_err_y = pred_velocity[:min_len, 1] - gt_vel[:min_len, 1]
rmse_vx = np.sqrt(np.mean(vel_err_x**2))
rmse_vy = np.sqrt(np.mean(vel_err_y**2))

step_lens = np.linalg.norm(np.diff(gt_pos[:min_len], axis=0), axis=1)
flight_length = np.sum(step_lens)
flight_time = (timestamps[min_len - 1] - timestamps[0]) / 1e9 if timestamps[0] > 1e3 else (timestamps[min_len - 1] - timestamps[0])

drift_pct = (final_drift_dist / flight_length) * 100 if flight_length > 0 else 0.0

print("=" * 55)
print("      ESKF DEAD RECKONING EVALUATION RESULTS")
print("=" * 55)
print(f"  Total RMSE (m)          : {rmse_pos:.4f}")
print(f"  Vx RMSE (m/s)           : {rmse_vx:.4f}")
print(f"  Vy RMSE (m/s)           : {rmse_vy:.4f}")
print(f"  ATE (m)                 : {ate:.4f}")
print(f"  Flight Length (m)       : {flight_length:.2f}")
print(f"  Flight Time (s)         : {flight_time:.2f}")
print(f"  Final Drift Dist (m)    : {final_drift_dist:.4f}")
print(f"  Drift %                 : {drift_pct:.2f} %")
print("=" * 55)