import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import os

class MocapKinematicDataset(Dataset):
    def __init__(self, dataset_folder, window_size=10):
        self.window_size = window_size
        
        imu_path = os.path.join(dataset_folder, "rs_imu.csv")
        mocap_path = os.path.join(dataset_folder, "mocap_vehicle_data.csv")
        
        imu_df = pd.read_csv(imu_path)
        mocap_df = pd.read_csv(mocap_path)
        
        # Strip whitespace from CSV headers
        imu_df.columns = [c.strip() for c in imu_df.columns]
        mocap_df.columns = [c.strip() for c in mocap_df.columns]
        
        imu_t = imu_df['t'].values
        mocap_t = mocap_df['t'].values
        
        # IMU Kinematic Features: Gyro(w_x, w_y, w_z), Accel(a_x, a_y, a_z)
        imu_features = imu_df[['w_x', 'w_y', 'w_z', 'a_x', 'a_y', 'a_z']].values
        
        # Extract vehicle positional states
        pos_x = mocap_df['p_x'].values
        pos_y = mocap_df['p_y'].values
        
        # Derive planar velocity states mechanically (dp/dt)
        vx_raw = np.gradient(pos_x, mocap_t)
        vy_raw = np.gradient(pos_y, mocap_t)
        
        # Apply Savitzky-Golay filter to reject high-frequency numerical noise
        vx_smooth = savgol_filter(vx_raw, window_length=15, polyorder=3)
        vy_smooth = savgol_filter(vy_raw, window_length=15, polyorder=3)
        
        # Align physical vehicle states to IMU sensor sampling rates
        interp_vx = interp1d(mocap_t, vx_smooth, bounds_error=False, fill_value="extrapolate")
        interp_vy = interp1d(mocap_t, vy_smooth, bounds_error=False, fill_value="extrapolate")
        interp_px = interp1d(mocap_t, pos_x - pos_x[0], bounds_error=False, fill_value="extrapolate")
        interp_py = interp1d(mocap_t, pos_y - pos_y[0], bounds_error=False, fill_value="extrapolate")
        
        aligned_vx = interp_vx(imu_t)
        aligned_vy = interp_vy(imu_t)
        aligned_px = interp_px(imu_t)
        aligned_py = interp_py(imu_t)
        
        # Filter for valid timestamps where both sensors capture data
        valid_idx = (imu_t >= mocap_t[0]) & (imu_t <= mocap_t[-1])
        
        self.imu_data = torch.tensor(imu_features[valid_idx], dtype=torch.float32)
        self.gt_vel_data = torch.tensor(np.column_stack((aligned_vx[valid_idx], aligned_vy[valid_idx])), dtype=torch.float32)
        self.gt_pos_data = torch.tensor(np.column_stack((aligned_px[valid_idx], aligned_py[valid_idx])), dtype=torch.float32)

    def __len__(self):
        return len(self.imu_data) - self.window_size

    def __getitem__(self, idx):
        x = self.imu_data[idx : idx + self.window_size]
        y = self.gt_vel_data[idx + self.window_size - 1]
        pos = self.gt_pos_data[idx + self.window_size - 1]
        return x, y, pos