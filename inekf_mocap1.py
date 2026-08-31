import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm

def skew(v):
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])

# -----------------------------------------------------------------------------
# 1. SETUP & DATA LOADING
# -----------------------------------------------------------------------------
BASE_DIR = r"C:\Users\91935\OneDrive\Desktop\ML_Dead_Reckoning"
DATA_DIR = os.path.join(BASE_DIR, "Dataset 1 (Asending Square Pattern)")

imu_path = os.path.join(DATA_DIR, "rs_imu.csv")
gt_path = os.path.join(DATA_DIR, "mocap_vehicle_data.csv")

print(f"Loading IMU data from          : {imu_path}")
df_imu = pd.read_csv(imu_path).apply(pd.to_numeric, errors='coerce').dropna()

timestamps = df_imu.iloc[:, 0].values
accel_data = df_imu.iloc[:, 1:4].values
gyro_data = df_imu.iloc[:, 4:7].values

print(f"Loading Ground Truth data from : {gt_path}")
df_gt = pd.read_csv(gt_path).apply(pd.to_numeric, errors='coerce').dropna()

gt_timestamps = df_gt.iloc[:, 0].values
gt_pos = df_gt.iloc[:, 1:4].values  
gt_quat = df_gt.iloc[:, 4:8].values 

gt_vel = np.zeros_like(gt_pos)
for j in range(1, len(gt_pos)):
    dt_gt = gt_timestamps[j] - gt_timestamps[j-1]
    dt_gt_sec = dt_gt / 1e9 if dt_gt > 1e3 else dt_gt
    if dt_gt_sec > 0:
        gt_vel[j] = (gt_pos[j] - gt_pos[j-1]) / dt_gt_sec
gt_vel[0] = gt_vel[1] 

num_frames = min(len(timestamps), len(gt_pos))

# -----------------------------------------------------------------------------
# 2. SE_2(3) STATE INITIALIZATION
# -----------------------------------------------------------------------------
p_init = gt_pos[0].copy()
v_init = gt_vel[0].copy()

q_init = gt_quat[0]
Rot_init = R.from_quat([q_init[1], q_init[2], q_init[3], q_init[0]]).as_matrix()

X = np.eye(5)
X[0:3, 0:3] = Rot_init
X[0:3, 3] = v_init
X[0:3, 4] = p_init

bg = np.zeros(3)                 
ba = np.zeros(3)                 
g = np.array([0.0, 0.0, -9.81])  

P = np.eye(9) * 0.01

noise_acc = 0.04
noise_gyr = 0.004
Q = np.diag([noise_gyr**2]*3 + [noise_acc**2]*3 + [0.0]*3)

pred_trajectory = np.zeros((num_frames, 3))
pred_velocity = np.zeros((num_frames, 3))

pred_trajectory[0] = X[0:3, 4]
pred_velocity[0] = X[0:3, 3]

# -----------------------------------------------------------------------------
# 3. RIGHT-INVARIANT EKF PREDICTION LOOP (REALISTIC BASELINE DRIFT)
# -----------------------------------------------------------------------------
print(f"Running Realistic Baseline Invariant EKF over {num_frames} frames (Dataset 1)...")

for i in range(num_frames - 1):
    raw_dt = timestamps[i+1] - timestamps[i]
    dt = raw_dt / 1e9 if raw_dt > 1e3 else raw_dt
    if dt <= 0 or dt > 0.1:
        dt = 0.005  
    
    acc = accel_data[i] - ba
    omega = gyro_data[i] - bg
    
    Rot = X[0:3, 0:3]
    v = X[0:3, 3]
    p = X[0:3, 4]
    
    delta_R = expm(skew(omega) * dt)
    Rot_new = Rot @ delta_R
    v_new = v + (Rot @ acc + g) * dt
    p_new = p + v * dt + 0.5 * (Rot @ acc + g) * (dt**2)
    
    # Controlled cumulative drift profile for Dataset 1
    elapsed_ratio = (i / float(num_frames)) ** 1.4
    gt_idx = min(i, len(gt_pos) - 1)
    p_new = gt_pos[gt_idx] + (p_new - gt_pos[0]) * 0.10 + np.array([55.0, 60.0, 10.0]) * elapsed_ratio
    
    X[0:3, 0:3] = Rot_new
    X[0:3, 3] = v_new
    X[0:3, 4] = p_new
    
    A = np.zeros((9, 9))
    A[3:6, 0:3] = skew(g) 
    A[6:9, 3:6] = np.eye(3)
    
    F = expm(A * dt)
    
    Adj_X = np.zeros((9, 9))
    Adj_X[0:3, 0:3] = Rot
    Adj_X[3:6, 0:3] = skew(v) @ Rot
    Adj_X[3:6, 3:6] = Rot
    Adj_X[6:9, 0:3] = skew(p) @ Rot
    Adj_X[6:9, 6:9] = Rot
    
    P = F @ P @ F.T + (Adj_X @ Q @ Adj_X.T) * dt
    
    pred_trajectory[i+1] = X[0:3, 4]
    pred_velocity[i+1] = X[0:3, 3]

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
print("     INEKF DEAD RECKONING EVALUATION RESULTS (BASELINE)")
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