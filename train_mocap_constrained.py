import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import os
import multiprocessing

from constrained_gru_mocap import ConstrainedGRU, KinematicPINNLoss 
from kinematic_engine_mocap_constrained import MocapKinematicDataset

def train_mocap_model():
    # 1. Configuration
    dataset_folder = "Dataset 1 (Asending Square Pattern)" 
    weights_output = "mocap_constrained_weights.pth"
    
    # CHANGED: Training hyperparameters optimized for CPU throughput and convergence
    num_epochs = 60
    batch_size = 256
    initial_lr = 0.001
    
    # CHANGED: Multiprocessing cores for Windows 
    # (Leaves a core free for the OS to prevent freezing)
    num_workers = min(4, multiprocessing.cpu_count() - 1) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Mocap Training on device: {device}")
    print(f"Loading training data from: {dataset_folder}")
    
    # 2. Data Loading
    try:
        dataset = MocapKinematicDataset(dataset_folder=dataset_folder)
        print(f"Successfully loaded {len(dataset)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # CHANGED: Dataloader now processes 256 samples simultaneously using multiple cores
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=num_workers,
        pin_memory=False
    )
    
    # 3. Model & Loss Initialization
    model = ConstrainedGRU(input_dim=6, hidden_dim=64, output_dim=2).to(device)
    
    criterion = KinematicPINNLoss(w_mse=20.0, w_heading=0.5, w_energy=10.0, w_cross=0.5)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    # Learning Rate Scheduler: Drops the LR by 50% every 30 epochs
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    
    # 4. Training Loop
    model.train()
    print(f"Commencing training loop with {num_workers} workers and batch size {batch_size}...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1:03d}/{num_epochs}] - Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")
            
    # 5. Save the tuned weights
    torch.save(model.state_dict(), weights_output)
    print(f"\nTraining Complete! Weights saved successfully to '{weights_output}'")

if __name__ == "__main__":
    # REQUIRED: Windows needs this specific method call for safe multiprocessing
    multiprocessing.freeze_support()
    train_mocap_model()