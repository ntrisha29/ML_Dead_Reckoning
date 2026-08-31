import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from constrained_gru import ConstrainedGRU
from kinematic_engine_constrained import EurocKinematicDataset

def apply_zero_phase_filter(signal, window_size=15):
    """
    Applies a zero-phase moving average filter to remove high-frequency prediction jitter.
    """
    if window_size < 2:
        return signal
    kernel = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, kernel, mode='same')
    return smoothed_signal

def evaluate_and_plot_constrained(dataset_folder, weights_path="euroc_constrained_weights.pth", dt=0.01, filter_window=15):
    """
    Evaluates model, applies filtering, uses Trapezoidal Integration for higher precision, 
    and generates visual plots.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Dataset directory '{dataset_folder}' not found.")
        
    dataset = EurocKinematicDataset(dataset_folder=dataset_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = ConstrainedGRU(input_dim=6, hidden_dim=128, output_dim=2).to(device)
    
    if os.path.exists(weights_path):
        print(f"Loading trained weights from: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        raise FileNotFoundError(f"Weights file '{weights_path}' not found! Run train script first.")
        
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
            
    gt_vx = np.array(gt_vx)
    gt_vy = np.array(gt_vy)
    pred_vx_raw = np.array(pred_vx_raw)
    pred_vy_raw = np.array(pred_vy_raw)

    # 1. Zero-Phase Filtering
    pred_vx = apply_zero_phase_filter(pred_vx_raw, window_size=filter_window)
    pred_vy = apply_zero_phase_filter(pred_vy_raw, window_size=filter_window)
    
    # 2. Trapezoidal Integration (Mathematically superior to Euler integration)
    # Calculates area under the velocity curve more accurately
    gt_x = np.concatenate(([0.0], np.cumsum((gt_vx[:-1] + gt_vx[1:]) / 2.0 * dt)))
    gt_y = np.concatenate(([0.0], np.cumsum((gt_vy[:-1] + gt_vy[1:]) / 2.0 * dt)))
    
    pred_x = np.concatenate(([0.0], np.cumsum((pred_vx[:-1] + pred_vx[1:]) / 2.0 * dt)))
    pred_y = np.concatenate(([0.0], np.cumsum((pred_vy[:-1] + pred_vy[1:]) / 2.0 * dt)))
        
    gt_traj = np.array([gt_x, gt_y]).T
    pred_traj = np.array([pred_x, pred_y]).T
    
    # 3. Compute Metrics
    final_drift_error = np.linalg.norm(gt_traj[-1] - pred_traj[-1])
    total_distance = np.sum(np.linalg.norm(np.diff(gt_traj, axis=0), axis=1))
    drift_percentage = (final_drift_error / total_distance) * 100 if total_distance > 0 else 0
    
    ate_errors = np.linalg.norm(gt_traj - pred_traj, axis=1)
    ate_rmse = np.sqrt(np.mean(ate_errors**2))
    
    mse_vx = np.mean((gt_vx - pred_vx)**2)
    mse_vy = np.mean((gt_vy - pred_vy)**2)
    
    print("\n" + "="*55)
    print("CONSTRAINED GRU EVALUATION RESULTS (EuRoC Dataset)")
    print("="*55)
    print(f"Dataset Folder            : {dataset_folder}")
    print(f"Weights Source            : {weights_path}")
    print(f"Filter Window Size        : {filter_window} steps")
    print(f"Integration Method        : Trapezoidal")
    print("-" * 55)
    print(f"Total Distance Traveled   : {total_distance:.4f} m")
    print(f"Final Position Drift      : {final_drift_error:.4f} m")
    print(f"Absolute Trajectory Error : {ate_rmse:.4f} m (ATE RMSE)")
    print(f"Integration Drift         : {drift_percentage:.2f}%")
    print("-" * 55)
    print(f"Velocity X (MSE)          : {mse_vx:.6f}")
    print(f"Velocity Y (MSE)          : {mse_vy:.6f}")
    print("="*55 + "\n")
    
    # 4. Plots
    plt.figure(1, figsize=(8, 8))
    plt.plot(gt_x, gt_y, label='Ground Truth', color='black', linestyle='--', linewidth=2)
    plt.plot(pred_x, pred_y, label='Constrained GRU (Trapezoidal)', color='blue', linewidth=2)
    plt.title('EuRoC: Dead Reckoning Trajectory (X vs Y)')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.figure(2, figsize=(10, 4))
    plt.plot(gt_vx, label='Ground Truth $V_x$', color='black', linestyle='--')
    plt.plot(pred_vx, label='Predicted $V_x$', color='blue', alpha=0.8)
    plt.title('EuRoC Kinematics: Velocity X over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)
    
    plt.figure(3, figsize=(10, 4))
    plt.plot(gt_vy, label='Ground Truth $V_y$', color='black', linestyle='--')
    plt.plot(pred_vy, label='Predicted $V_y$', color='blue', alpha=0.8)
    plt.title('EuRoC Kinematics: Velocity Y over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    DATASET_PATH = "EuRoC MAV dataset"
    WEIGHTS_PATH = "euroc_constrained_weights.pth"
    
    evaluate_and_plot_constrained(
        dataset_folder=DATASET_PATH, 
        weights_path=WEIGHTS_PATH, 
        dt=0.01,
        filter_window=15
    )