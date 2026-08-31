import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class MocapKinematicDataset(Dataset):
    # CHANGED: seq_len adjusted to 50 for CPU optimization
    def __init__(self, dataset_folder, dt=0.01, seq_len=50):
        """
        Loads IMU and Mocap CSV files. Computes Ground Truth velocity 
        via numerical differentiation, normalizes IMU features, and generates 
        sliding windows to provide temporal context for recurrent neural networks.
        """
        self.seq_len = seq_len
        
        imu_path = os.path.join(dataset_folder, "rs_imu.csv")
        mocap_path = os.path.join(dataset_folder, "mocap_vehicle_data.csv")
        
        if not os.path.exists(imu_path) or not os.path.exists(mocap_path):
            raise FileNotFoundError(f"Missing CSV files in {dataset_folder}. Ensure rs_imu.csv and mocap_vehicle_data.csv exist.")

        # 1. Load CSVs
        imu_df = pd.read_csv(imu_path)
        mocap_df = pd.read_csv(mocap_path)
        
        # Truncate to the shortest array to align the datasets
        min_length = min(len(imu_df), len(mocap_df))
        
        # 2. Extract and Normalize IMU Features (Inputs)
        # Slicing columns B through G (Indices 1 to 6) -> a_x, a_y, a_z, w_x, w_y, w_z
        imu_features = imu_df.iloc[:min_length, 1:7].values 
        
        # Standardize IMU data (Mean=0, Variance=1)
        scaler = StandardScaler()
        imu_features = scaler.fit_transform(imu_features)
        
        # 3. Extract Mocap Position & Compute Velocity (Targets)
        p_x = mocap_df.iloc[:min_length, 1].values 
        p_y = mocap_df.iloc[:min_length, 2].values 
        
        # Numerically differentiate position to get velocity
        v_x = np.gradient(p_x) / dt
        v_y = np.gradient(p_y) / dt
        
        mocap_targets = np.column_stack((v_x, v_y))
        
        # 4. Convert to PyTorch Tensors
        self.inputs = torch.tensor(imu_features, dtype=torch.float32)
        self.targets = torch.tensor(mocap_targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.inputs) - self.seq_len + 1
        
    def __getitem__(self, idx):
        window_inputs = self.inputs[idx : idx + self.seq_len]
        target_velocity = self.targets[idx + self.seq_len - 1]
        
        return window_inputs, target_velocity

if __name__ == "__main__":
    test_folder = "Dataset 1 (Asending Square Pattern)"
    if os.path.exists(test_folder):
        print("Testing Dataset Engine...")
        dataset = MocapKinematicDataset(test_folder, seq_len=50)
        inputs, targets = dataset[0]
        print(f"Dataset successfully loaded {len(dataset)} windowed samples.")
        print(f"Input Shape : {inputs.shape} (Expected: 50, 6)")
        print(f"Target Shape: {targets.shape} (Expected: 2)")
    else:
        print(f"Could not find '{test_folder}' to run test.")