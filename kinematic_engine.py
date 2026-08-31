import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d

class KinematicDataset(Dataset):
    def __init__(self, imu_path, gt_path, window_size=10):
        self.window_size = window_size
        
        # 1. Load the raw MH_01 CSV files
        imu_df = pd.read_csv(imu_path)
        gt_df = pd.read_csv(gt_path)

        # 2. Extract Timestamps (in nanoseconds)
        imu_t = imu_df.iloc[:, 0].values
        gt_t = gt_df.iloc[:, 0].values
        
        # 3. Extract Features (IMU 6-DoF)
        # Columns: Gyro(X,Y,Z), Accel(X,Y,Z)
        imu_features = imu_df.iloc[:, 1:7].values
        
        # 4. Extract Ground Truth Targets
        # GT Columns: Pos(X,Y,Z), Quat(W,X,Y,Z), Vel(X,Y,Z), Biases...
        # We want Pos X/Y and Vel X/Y
        gt_pos = gt_df.iloc[:, 1:3].values # p_RS_R_x, p_RS_R_y
        gt_vel = gt_df.iloc[:, 8:10].values # v_RS_R_x, v_RS_R_y
        
        # Zero-center the Ground Truth Positions (Shift origin to 0,0)
        gt_pos = gt_pos - gt_pos[0, :]
        
        # 5. Timestamp Synchronization
        # The IMU and GT cameras capture at slightly different nanoseconds. 
        # We interpolate the GT velocities and positions to match the exact IMU timestamps.
        interp_vel = interp1d(gt_t, gt_vel, axis=0, bounds_error=False, fill_value="extrapolate")
        interp_pos = interp1d(gt_t, gt_pos, axis=0, bounds_error=False, fill_value="extrapolate")
        
        aligned_gt_vel = interp_vel(imu_t)
        aligned_gt_pos = interp_pos(imu_t)
        
        # We only keep timestamps where both sensors were actively recording
        valid_idx = (imu_t >= gt_t[0]) & (imu_t <= gt_t[-1])
        self.imu_data = imu_features[valid_idx]
        self.gt_vel_data = aligned_gt_vel[valid_idx]
        self.gt_pos_data = aligned_gt_pos[valid_idx] # Kept for plotting later
        
        # Convert to float32 tensors for PyTorch
        self.imu_data = torch.tensor(self.imu_data, dtype=torch.float32)
        self.gt_vel_data = torch.tensor(self.gt_vel_data, dtype=torch.float32)
        self.gt_pos_data = torch.tensor(self.gt_pos_data, dtype=torch.float32)

    def __len__(self):
        return len(self.imu_data) - self.window_size

    def __getitem__(self, idx):
        # Return a sliding window of IMU data and the TARGET velocity at the END of the window
        x = self.imu_data[idx : idx + self.window_size]
        y = self.gt_vel_data[idx + self.window_size - 1]
        pos = self.gt_pos_data[idx + self.window_size - 1]
        return x, y, pos