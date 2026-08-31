import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from kinematic_engine import KinematicDataset
import os

class UnconstrainedGRU(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=2):
        super(UnconstrainedGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

if __name__ == "__main__":
    print("Initializing datasets...")
    
    # Path configuration - assuming data is in the EuRoC folder
    imu_path = os.path.join("EuRoC MAV dataset", "imu_data.csv")
    gt_path = os.path.join("EuRoC MAV dataset", "gt_data.csv")
    
    dataset = KinematicDataset(imu_path=imu_path, gt_path=gt_path)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}\n")
    
    model = UnconstrainedGRU().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    print("Starting Training...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Notice we unpack three variables now (inputs, targets_vel, targets_pos)
        for inputs, targets_vel, _ in train_loader:
            inputs, targets_vel = inputs.to(device), targets_vel.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets_vel)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
        
    print("Training test complete!")