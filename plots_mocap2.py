import torch
import matplotlib.pyplot as plt
from unconstrained_gru_mocap import train_model

# We import the exact same evaluation and plotting kinematics we built for the square pattern
from plots_mocap1 import evaluate_and_plot

if __name__ == "__main__":
    # Point the pipeline to the new dataset directory
    folder = "Dataset 2 (Upward Spiral Pattern)"
    
    # Train the baseline GRU on the spiral vehicle dynamics
    model, dataset, device = train_model(folder)
    
    # Generate the metrics and overlay plots
    evaluate_and_plot(model, dataset, device, "Upward Spiral Pattern")