import os
import glob
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. LOAD DATASETS
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pred_path = os.path.join(BASE_DIR, "gtsam_euroc_trajectory.csv")

EUROC_DIR = os.path.join(BASE_DIR, "EuRoC MAV dataset")
gt_files = glob.glob(os.path.join(EUROC_DIR, "**", "*.csv"), recursive=True)
gt_path = gt_files[0] if gt_files else os.path.join(EUROC_DIR, "gt_data.csv")

pred_df = pd.read_csv(pred_path)
gt_df = pd.read_csv(gt_path)

# Extract XYZ positions
pred_pos = pred_df[["x", "y", "z"]].values
gt_data = gt_df.apply(pd.to_numeric, errors='coerce').dropna().values

# Align array lengths
min_len = min(len(pred_pos), len(gt_data))
pred_pos = pred_pos[:min_len]
gt_pos = gt_data[:min_len, 1:4] if gt_data.shape[1] >= 4 else gt_data[:min_len, :3]

# -----------------------------------------------------------------------------
# 2. METRIC CALCULATIONS
# -----------------------------------------------------------------------------
# Absolute Trajectory Error (ATE)
position_errors = np.linalg.norm(pred_pos - gt_pos, axis=1)
ate = np.sqrt(np.mean(position_errors ** 2))

# Velocity RMSE (Numerical derivative of position errors)
dt = 0.005  # 200 Hz default timestep for EuRoC
pred_vel = np.diff(pred_pos, axis=0) / dt
gt_vel = np.diff(gt_pos, axis=0) / dt
vel_errors = np.linalg.norm(pred_vel - gt_vel, axis=1)
rmse_vel = np.sqrt(np.mean(vel_errors ** 2))

# Trajectory Drift Percentage
step_lengths = np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)
total_path_length = np.sum(step_lengths)
final_drift_error = position_errors[-1]
drift_percent = (final_drift_error / total_path_length) * 100

# -----------------------------------------------------------------------------
# 3. DISPLAY RESULTS
# -----------------------------------------------------------------------------
print("=" * 55)
print("  GTSAM CLASSICAL PREINTEGRATION BENCHMARK RESULTS")
print("=" * 55)
print(f"Velocity RMSE (m/s) : {rmse_vel:.4f}")
print(f"ATE (m)            : {ate:.4f}")
print(f"Drift (%)          : {drift_percent:.2f}%")
print("=" * 55)