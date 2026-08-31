import pandas as pd
import matplotlib.pyplot as plt

def plot_euroc_trajectories(csv_path):
    print(f"Loading Ground Truth data from: {csv_path}")
    
    # Load dataset
    df = pd.read_csv("C:\\Users\\91935\\OneDrive\\Desktop\\ML_Dead_Reckoning\\EuRoC MAV dataset\\gt_data.csv")
    
    # Strip any leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()

    # Extract positional data
    x = df['p_RS_R_x [m]']
    y = df['p_RS_R_y [m]']
    z = df['p_RS_R_z [m]']

    # ---------------------------------------------------------
    # 1. Plot 2D Trajectory (XY Plane)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='Ground Truth Path', color='black', linewidth=1.5)
    plt.scatter(x.iloc[0], y.iloc[0], color='green', label='Start', zorder=5)
    plt.scatter(x.iloc[-1], y.iloc[-1], color='red', label='End', zorder=5)
    
    # Formatting
    plt.title('EuRoC MAV Ground Truth Trajectory (XY Plane)', fontsize=14, fontweight='bold')
    plt.xlabel('Position X [m]', fontsize=12)
    plt.ylabel('Position Y [m]', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save and show
    plt.savefig('euroc_2d_trajectory.png', dpi=300)
    print("Saved 2D plot as 'euroc_2d_trajectory.png'")

    # ---------------------------------------------------------
    # 2. Plot 3D Trajectory
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x, y, z, label='Ground Truth Path', color='blue', linewidth=1.5)
    ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], color='green', label='Start', s=50, zorder=5)
    ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], color='red', label='End', s=50, zorder=5)
    
    # Formatting
    ax.set_title('EuRoC MAV Ground Truth Trajectory (3D)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Position X [m]', labelpad=10, fontsize=12)
    ax.set_ylabel('Position Y [m]', labelpad=10, fontsize=12)
    ax.set_zlabel('Position Z [m]', labelpad=10, fontsize=12)
    ax.legend(loc='best')
    plt.tight_layout()
    
    # Save and show
    plt.savefig('euroc_3d_trajectory.png', dpi=300)
    print("Saved 3D plot as 'euroc_3d_trajectory.png'")

if __name__ == "__main__":
    plot_euroc_trajectories('gt_data.csv')