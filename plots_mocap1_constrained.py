import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from constrained_gru_mocap import ConstrainedGRU
from kinematic_engine_mocap_constrained import MocapKinematicDataset

def apply_zero_phase_filter(signal, window_size=15):
    if window_size < 2:
        return signal
    kernel = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, kernel, mode='same')
    return smoothed_signal

def evaluate_mocap1():
    dataset_folder = "Dataset 1 (Asending Square Pattern)"
    weights_path = "mocap_constrained_weights.pth" 
    dt = 0.01
    filter_window = 15
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Mocap Dataset 1 on {device}...")
    
    dataset = MocapKinematicDataset(dataset_folder=dataset_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = ConstrainedGRU(input_dim=6, hidden_dim=64, output_dim=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    gt_vx, gt_vy = [], []
    pred_vx_raw, pred_vy_raw = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            v_pred = model(batch_x)
            
            pred_vx_raw.append(v_pred[0, 0].item())
            pred_vy_raw.append(v_pred[0, 1].item())
            gt_vx.append(batch_y[0, 0].item())
            gt_vy.append(batch_y[0, 1].item())
            
    gt_vx, gt_vy = np.array(gt_vx), np.array(gt_vy)
    pred_vx_raw, pred_vy_raw = np.array(pred_vx_raw), np.array(pred_vy_raw)

    pred_vx = apply_zero_phase_filter(pred_vx_raw, window_size=filter_window)
    pred_vy = apply_zero_phase_filter(pred_vy_raw, window_size=filter_window)
    
    # --- ZERO-VELOCITY UPDATE (ZUPT) DEADBAND ---
    # Snaps micro-velocities to zero to prevent integration drift during hover
    noise_floor_threshold = 0.03 # m/s
    pred_vx[np.abs(pred_vx) < noise_floor_threshold] = 0.0
    pred_vy[np.abs(pred_vy) < noise_floor_threshold] = 0.0
    
    # Trapezoidal Integration for Position
    gt_x = np.concatenate(([0.0], np.cumsum((gt_vx[:-1] + gt_vx[1:]) / 2.0 * dt)))
    gt_y = np.concatenate(([0.0], np.cumsum((gt_vy[:-1] + gt_vy[1:]) / 2.0 * dt)))
    pred_x = np.concatenate(([0.0], np.cumsum((pred_vx[:-1] + pred_vx[1:]) / 2.0 * dt)))
    pred_y = np.concatenate(([0.0], np.cumsum((pred_vy[:-1] + pred_vy[1:]) / 2.0 * dt)))
        
    gt_traj = np.array([gt_x, gt_y]).T
    pred_traj = np.array([pred_x, pred_y]).T
    
    # --- METRICS CALCULATION ---
    final_drift_error = np.linalg.norm(gt_traj[-1] - pred_traj[-1])
    total_distance = np.sum(np.linalg.norm(np.diff(gt_traj, axis=0), axis=1))
    drift_percentage = (final_drift_error / total_distance) * 100 if total_distance > 0 else 0
    ate_rmse = np.sqrt(np.mean(np.linalg.norm(gt_traj - pred_traj, axis=1)**2))
    
    rmse_vx = np.sqrt(np.mean((gt_vx - pred_vx)**2))
    rmse_vy = np.sqrt(np.mean((gt_vy - pred_vy)**2))
    
    # --- TERMINAL OUTPUT ---
    print("\n" + "="*55)
    print("DATASET 1: ASCENDING SQUARE PATTERN")
    print("="*55)
    print(f"Total Distance Traveled   : {total_distance:.4f} m")
    print(f"Final Position Drift      : {final_drift_error:.4f} m")
    print(f"Absolute Trajectory Error : {ate_rmse:.4f} m")
    print(f"Integration Drift         : {drift_percentage:.2f}%")
    print("-" * 55)
    print(f"Velocity RMSE (X-axis)    : {rmse_vx:.4f} m/s")
    print(f"Velocity RMSE (Y-axis)    : {rmse_vy:.4f} m/s")
    print("="*55 + "\n")
    
    # --- PLOTTING ---
    time_axis = np.arange(len(gt_x)) * dt
    time_axis_vel = np.arange(len(gt_vx)) * dt

    plt.figure(1, figsize=(8, 8))
    plt.plot(gt_x, gt_y, label='Ground Truth', color='black', linestyle='--')
    plt.plot(pred_x, pred_y, label='PINN Prediction', color='blue')
    plt.title('Dataset 1: Ascending Square (X vs Y Trajectory)')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    fig_v, (ax_vx, ax_vy) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_vx.plot(time_axis_vel, gt_vx, label='Ground Truth', color='black', linestyle='--')
    ax_vx.plot(time_axis_vel, pred_vx, label='Prediction', color='red', alpha=0.7)
    ax_vx.set_title('Velocity Tracking - X Axis')
    ax_vx.set_ylabel('Velocity (m/s)')
    ax_vx.legend()
    ax_vx.grid(True)
    
    ax_vy.plot(time_axis_vel, gt_vy, label='Ground Truth', color='black', linestyle='--')
    ax_vy.plot(time_axis_vel, pred_vy, label='Prediction', color='green', alpha=0.7)
    ax_vy.set_title('Velocity Tracking - Y Axis')
    ax_vy.set_xlabel('Time (s)')
    ax_vy.set_ylabel('Velocity (m/s)')
    ax_vy.legend()
    ax_vy.grid(True)

    fig_p, (ax_px, ax_py) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_px.plot(time_axis, gt_x, label='Ground Truth', color='black', linestyle='--')
    ax_px.plot(time_axis, pred_x, label='Prediction', color='blue', alpha=0.7)
    ax_px.set_title('Position Tracking - X Axis')
    ax_px.set_ylabel('Position (m)')
    ax_px.legend()
    ax_px.grid(True)
    
    ax_py.plot(time_axis, gt_y, label='Ground Truth', color='black', linestyle='--')
    ax_py.plot(time_axis, pred_y, label='Prediction', color='purple', alpha=0.7)
    ax_py.set_title('Position Tracking - Y Axis')
    ax_py.set_xlabel('Time (s)')
    ax_py.set_ylabel('Position (m)')
    ax_py.legend()
    ax_py.grid(True)
    
    plt.show()

if __name__ == "__main__":
    evaluate_mocap1()