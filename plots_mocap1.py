import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from unconstrained_gru_mocap import train_model

def evaluate_and_plot(model, dataset, device, dataset_name):
    model.eval()
    
    all_preds = []
    all_targets_vel = []
    all_targets_pos = []
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    with torch.no_grad():
        for inputs, targets_vel, targets_pos in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets_vel.append(targets_vel.numpy())
            all_targets_pos.append(targets_pos.numpy())
            
    preds = np.vstack(all_preds)
    targets_vel = np.vstack(all_targets_vel)
    gt_pos = np.vstack(all_targets_pos)
    
    # --- METRICS ---
    rmse_vx = np.sqrt(np.mean((preds[:, 0] - targets_vel[:, 0])**2))
    rmse_vy = np.sqrt(np.mean((preds[:, 1] - targets_vel[:, 1])**2))
    total_rmse = np.sqrt(np.mean((preds - targets_vel)**2))
    
    # Integrate predictions to get trajectory (dt = 0.005 for 200Hz MoCap IMU)
    dt = 0.005 
    pred_pos_x = np.cumsum(preds[:, 0] * dt)
    pred_pos_y = np.cumsum(preds[:, 1] * dt)
    pred_pos = np.column_stack((pred_pos_x, pred_pos_y))
    
    # Calculate ATE and Drift using the real GT positions
    position_errors = np.linalg.norm(pred_pos - gt_pos, axis=1)
    ate = np.mean(position_errors)
    final_drift = position_errors[-1]
    
    gt_step_distances = np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)
    total_gt_distance = np.sum(gt_step_distances)
    drift_percentage = (final_drift / total_gt_distance) * 100 if total_gt_distance > 0 else 0

    print("\n" + "="*40)
    print(f"FINAL EVALUATION METRICS ({dataset_name}):")
    print("="*40)
    print(f"Velocity RMSE (Total) : {total_rmse:.4f} m/s")
    print(f"  - RMSE (Vx)         : {rmse_vx:.4f} m/s")
    print(f"  - RMSE (Vy)         : {rmse_vy:.4f} m/s")
    print(f"Absolute Traj. (ATE)  : {ate:.4f} m")
    print(f"Total Flight Length   : {total_gt_distance:.4f} m")
    print(f"Final Drift           : {final_drift:.4f} m")
    print(f"Drift Percentage      : {drift_percentage:.2f} %")
    print("="*40 + "\n")

    # --- PLOTS ---
    # Plot 1: XY Trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], label=f"Ground Truth ({dataset_name})", color='black', linewidth=1.5)
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], label="Baseline (Pure IMU)", color='blue', linestyle=':', alpha=0.7)
    plt.scatter(gt_pos[0, 0], gt_pos[0, 1], color='green', marker='o', s=100, zorder=5, label='Start')
    plt.scatter(gt_pos[-1, 0], gt_pos[-1, 1], color='red', marker='X', s=100, zorder=5, label='End (GT)')
    plt.title(f"{dataset_name} Trajectory Overlay - XY Plane")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Plot 2: Velocity Tracking
    time_steps = np.arange(len(preds)) * dt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(time_steps, targets_vel[:, 0], label="GT Vx", color='black', alpha=0.8)
    ax1.plot(time_steps, preds[:, 0], label="Pred Vx", color='blue', alpha=0.6)
    ax1.set_title(f"{dataset_name}: Velocity X ($V_x$) over Time")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(time_steps, targets_vel[:, 1], label="GT Vy", color='black', alpha=0.8)
    ax2.plot(time_steps, preds[:, 1], label="Pred Vy", color='red', alpha=0.6)
    ax2.set_title(f"{dataset_name}: Velocity Y ($V_y$) over Time")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()

    # Plot 3: Residuals
    plt.figure(figsize=(8, 6))
    plt.hist(preds[:, 0] - targets_vel[:, 0], bins=50, alpha=0.5, label='Vx Residuals', color='blue')
    plt.hist(preds[:, 1] - targets_vel[:, 1], bins=50, alpha=0.5, label='Vy Residuals', color='red')
    plt.axvline(0, color='black', linestyle='dashed', linewidth=1)
    plt.title(f"Error Residual Distribution ({dataset_name})")
    plt.xlabel("Prediction Error (m/s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    folder = "Dataset 1 (Asending Square Pattern)"
    model, dataset, device = train_model(folder)
    evaluate_and_plot(model, dataset, device, "Ascending Square Pattern")