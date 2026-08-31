import torch
from constrained_gru import train_model

def main():
    # 1. Point to your specific dataset folder
    dataset_folder = "EuRoC MAV dataset"
    
    # 2. Train the mechanically constrained model
    # (Using 20 epochs as a solid baseline for kinematics)
    print("Initializing training for EuRoC constrained system...")
    trained_model, _, _ = train_model(dataset_folder=dataset_folder, epochs=20, batch_size=32)
    
    # 3. Save the learned physical weights
    save_path = "euroc_constrained_weights.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"\nModel weights successfully saved to: {save_path}")

if __name__ == "__main__":
    main()