import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from kinematic_engine_mocap2_constrained import MocapDataset, KinematicPINNLoss

class ConstrainedGRU(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, output_size=2):
        super(ConstrainedGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        # Take the output of the last time step
        out = self.fc1(out[:, -1, :])
        return out

def train_mocap2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Mocap Training on device: {device}")
    
    # File paths - UPDATE THESE TO YOUR LOCAL PATHS IF DIFFERENT
    imu_path = 'C:\\Users\\91935\\OneDrive\\Desktop\\ML_Dead_Reckoning\\Dataset 2 (Upward Spiral Pattern)\\rs_imu.csv'
    gt_path = 'C:\\Users\\91935\\OneDrive\\Desktop\\ML_Dead_Reckoning\\Dataset 2 (Upward Spiral Pattern)\\mocap_vehicle_data.csv'
    
    dataset = MocapDataset(imu_path, gt_path, seq_len=50)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    
    model = ConstrainedGRU(input_size=6, hidden_size=128, output_size=2).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = KinematicPINNLoss(w_energy=2.0, zero_vel_threshold=0.03)
    
    epochs = 90
    print(f"Commencing training loop with 4 workers and batch size 256...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred_y = model(batch_x)
            loss = criterion(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1:03d}/{epochs}] - Loss: {total_loss/len(dataloader):.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    torch.save(model.state_dict(), 'mocap2_constrained_weights.pth')
    print("Training Complete! Weights saved successfully to 'mocap2_constrained_weights.pth'")

if __name__ == "__main__":
    train_mocap2()