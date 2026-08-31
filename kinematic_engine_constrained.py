import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

class EurocKinematicDataset(Dataset):
    """
    Data loader for the EuRoC MAV dataset.
    Expects 'imu_data.csv' and 'gt_data.csv' in the target directory.
    """
    def __init__(self, dataset_folder, seq_length=20):
        self.seq_length = seq_length
        
        imu_path = os.path.join(dataset_folder, 'imu_data.csv')
        gt_path = os.path.join(dataset_folder, 'gt_data.csv')
        
        if not os.path.exists(imu_path) or not os.path.exists(gt_path):
            raise FileNotFoundError(f"Missing CSV files in {dataset_folder}. Check folder path.")

        # Load data
        self.imu_df = pd.read_csv(imu_path)
        self.gt_df = pd.read_csv(gt_path)
        
        self.x_data = self.imu_df.iloc[:, 1:7].values 
        self.y_data = self.gt_df.iloc[:, 1:3].values  

        # --- THE FIX: Dataset Alignment ---
        # Find the minimum length to prevent out-of-bounds indexing if sensors stopped at different times
        self.valid_len = min(len(self.x_data), len(self.y_data))
        
        # Truncate both arrays to the exact same length
        self.x_data = self.x_data[:self.valid_len]
        self.y_data = self.y_data[:self.valid_len]

        # Normalize IMU data (Standardization)
        self.x_data = (self.x_data - np.mean(self.x_data, axis=0)) / np.std(self.x_data, axis=0)

    def __len__(self):
        return self.valid_len - self.seq_length

    def __getitem__(self, idx):
        x_window = self.x_data[idx : idx + self.seq_length]
        y_target = self.y_data[idx + self.seq_length - 1]
        
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)