import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class MocapDataset(Dataset):
    def __init__(self, imu_path, gt_path, seq_len=50):
        self.seq_len = seq_len
        
        # Load raw datasets
        imu_df = pd.read_csv(imu_path)
        gt_df = pd.read_csv(gt_path)
        
        # Strip whitespaces to prevent KeyError
        imu_df.columns = imu_df.columns.str.strip()
        gt_df.columns = gt_df.columns.str.strip()
        
        # Merge datasets on timestamps to align IMU with Ground Truth
        imu_df = imu_df.sort_values('t')
        gt_df = gt_df.sort_values('t')
        df = pd.merge_asof(imu_df, gt_df, on='t', direction='nearest')
        
        # Calculate time differential (dt) for integration/differentiation
        self.dt = np.diff(df['t'].values, prepend=df['t'].values[0])
        self.dt[0] = self.dt[1] if len(self.dt) > 1 else 0.01 
        
        # Numerically differentiate position to derive ground truth velocity (X and Y)
        v_x = np.gradient(df['p_x'].values, df['t'].values)
        v_y = np.gradient(df['p_y'].values, df['t'].values)
        self.target_vel = np.stack([v_x, v_y], axis=1)
        
        # Extract IMU features (Accelerometer & Gyroscope)
        imu_features = df[['a_x', 'a_y', 'a_z', 'w_x', 'w_y', 'w_z']].values
        
        # Standardize IMU inputs to prevent exploding gradients
        self.scaler = StandardScaler()
        self.imu_scaled = self.scaler.fit_transform(imu_features)
        
        # Store positions and timestamps for evaluation mapping
        self.gt_pos = df[['p_x', 'p_y']].values
        self.time = df['t'].values

    def __len__(self):
        return len(self.imu_scaled) - self.seq_len

    def __getitem__(self, idx):
        # Sliding window for sequential GRU context
        x = self.imu_scaled[idx : idx + self.seq_len]
        # Target velocity maps to the end of the sliding window
        y = self.target_vel[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class KinematicPINNLoss(nn.Module):
    def __init__(self, w_energy=2.0, zero_vel_threshold=0.03):
        super(KinematicPINNLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.w_energy = w_energy
        self.zero_vel_threshold = zero_vel_threshold

    def forward(self, pred_v, target_v):
        # 1. Base Mean Squared Error
        loss_mse = self.mse(pred_v, target_v)
        
        # 2. ZUPT (Zero-Velocity Update) Deadband Constraint
        target_mag = torch.norm(target_v, dim=1)
        stationary_mask = (target_mag < self.zero_vel_threshold).float()
        pred_mag = torch.norm(pred_v, dim=1)
        loss_zupt = torch.mean(stationary_mask * (pred_mag ** 2))
        
        # 3. Kinematic Magnitude Constraint
        loss_energy = torch.mean((pred_mag - target_mag) ** 2)
        
        return loss_mse + (self.w_energy * loss_energy) + loss_zupt

def evaluate_trajectory(model, dataset, device, title="DATASET 2: UPWARD SPIRAL PATTERN"):
    model.eval()
    
    pred_velocities = []
    gt_velocities = []
    
    # Execute predictions sequentially without shuffling
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device) 
            pred_v = model(x).squeeze(0).cpu().numpy()
            
            pred_velocities.append(pred_v)
            gt_velocities.append(y.numpy())
            
    pred_velocities = np.array(pred_velocities)
    gt_velocities = np.array(gt_velocities)
    
    # Align arrays based on the sequence length offset
    gt_pos = dataset.gt_pos[dataset.seq_len:]
    dt_array = dataset.dt[dataset.seq_len:]
    time_array = dataset.time[dataset.seq_len:]
    time_array = time_array - time_array[0] 
    
    # Integrate predicted velocity to estimate position (Dead Reckoning)
    pred_pos = np.zeros_like(gt_pos)
    pred_pos[0] = gt_pos[0] 
    
    for i in range(1, len(pred_velocities)):
        dt = dt_array[i]
        pred_pos[i, 0] = pred_pos[i-1, 0] + pred_velocities[i, 0] * dt
        pred_pos[i, 1] = pred_pos[i-1, 1] + pred_velocities[i, 1] * dt
        
    # Terminal Metrics Calculation
    total_distance = np.sum(np.sqrt(np.sum(np.diff(gt_pos, axis=0)**2, axis=1)))
    final_drift = np.linalg.norm(pred_pos[-1] - gt_pos[-1])
    ate = np.sqrt(np.mean(np.sum((pred_pos - gt_pos)**2, axis=1)))
    integration_drift_pct = (final_drift / total_distance) * 100 if total_distance > 0 else 0
    
    rmse_v_x = np.sqrt(np.mean((pred_velocities[:, 0] - gt_velocities[:, 0])**2))
    rmse_v_y = np.sqrt(np.mean((pred_velocities[:, 1] - gt_velocities[:, 1])**2))
    
    print("\n==================================================================")
    print(f"{title}")
    print("==================================================================")
    print(f"Total Distance Traveled   : {total_distance:.4f} m")
    print(f"Final Position Drift      : {final_drift:.4f} m")
    print(f"Absolute Trajectory Error : {ate:.4f} m")
    print(f"Integration Drift         : {integration_drift_pct:.2f}%")
    print("------------------------------------------------------------------")
    print(f"Velocity RMSE (X-axis)    : {rmse_v_x:.4f} m/s")
    print(f"Velocity RMSE (Y-axis)    : {rmse_v_y:.4f} m/s")
    print("==================================================================\n")
    
    # Plot 1: X vs Y Trajectory Map
    plt.figure(figsize=(8, 8))
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'k--', label='Ground Truth')
    plt.plot(pred_pos[:, 0], pred_pos[:, 1], 'b-', label='PINN Prediction')
    plt.title(f'{title} (X vs Y Trajectory)')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
    # Plot 2: Velocity Tracking Time-Series
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(time_array, gt_velocities[:, 0], 'k--', label='Ground Truth')
    ax1.plot(time_array, pred_velocities[:, 0], 'r-', label='Prediction', alpha=0.7)
    ax1.set_title('Velocity Tracking - X Axis')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(time_array, gt_velocities[:, 1], 'k--', label='Ground Truth')
    ax2.plot(time_array, pred_velocities[:, 1], 'g-', label='Prediction', alpha=0.7)
    ax2.set_title('Velocity Tracking - Y Axis')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.legend()
    ax2.grid(True)
    plt.show()
    
    # Plot 3: Position Tracking Time-Series
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(time_array, gt_pos[:, 0], 'k--', label='Ground Truth')
    ax1.plot(time_array, pred_pos[:, 0], 'b-', label='Prediction', alpha=0.7)
    ax1.set_title('Position Tracking - X Axis')
    ax1.set_ylabel('Position (m)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(time_array, gt_pos[:, 1], 'k--', label='Ground Truth')
    ax2.plot(time_array, pred_pos[:, 1], 'purple', label='Prediction', alpha=0.7)
    ax2.set_title('Position Tracking - Y Axis')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.legend()
    ax2.grid(True)
    plt.show()