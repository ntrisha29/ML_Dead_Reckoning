import torch
import torch.nn as nn
from kinematic_engine_mocap2_constrained import MocapDataset, evaluate_trajectory

class ConstrainedGRU(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, output_size=2):
        super(ConstrainedGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        # Back to a single layer to match your newly saved weights
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc1(out[:, -1, :])
        return out

def evaluate_mocap2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating Mocap Dataset 2 on {device}...")
    
    # File paths
    imu_path = 'C:\\Users\\91935\\OneDrive\\Desktop\\ML_Dead_Reckoning\\Dataset 2 (Upward Spiral Pattern)\\rs_imu.csv'
    gt_path = 'C:\\Users\\91935\\OneDrive\\Desktop\\ML_Dead_Reckoning\\Dataset 2 (Upward Spiral Pattern)\\mocap_vehicle_data.csv'
    weights_path = 'mocap2_constrained_weights.pth'
    
    dataset = MocapDataset(imu_path, gt_path, seq_len=50)
    
    # Initialize the 128-node model to match the training architecture
    model = ConstrainedGRU(input_size=6, hidden_size=128, output_size=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    evaluate_trajectory(model, dataset, device, title="DATASET 2: UPWARD SPIRAL PATTERN")

if __name__ == "__main__":
    evaluate_mocap2()