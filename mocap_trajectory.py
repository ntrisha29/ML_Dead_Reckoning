import pandas as pd
import matplotlib.pyplot as plt

def plot_synthetic_profiles():
    # Load datasets and clean column names using the exact folder paths
    df1 = pd.read_csv(r'C:\Users\91935\OneDrive\Desktop\ML_Dead_Reckoning\Dataset 1 (Asending Square Pattern)\mocap_vehicle_data.csv')
    df1.columns = df1.columns.str.strip()

    df2 = pd.read_csv(r'C:\Users\91935\OneDrive\Desktop\ML_Dead_Reckoning\Dataset 2 (Upward Spiral Pattern)\mocap_vehicle_data_2.csv')
    df2.columns = df2.columns.str.strip()

    # Create a figure with two 3D subplots side by side
    fig = plt.figure(figsize=(16, 7))

    # ---------------------------------------------------------
    # Subplot 1: Ascending Square Pattern
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(df1['p_x'], df1['p_y'], df1['p_z'], label='Ground Truth Path', color='blue', linewidth=1.5)

    # Mark Start and End
    ax1.scatter(df1['p_x'].iloc[0], df1['p_y'].iloc[0], df1['p_z'].iloc[0], color='green', label='Start', s=60, zorder=5)
    ax1.scatter(df1['p_x'].iloc[-1], df1['p_y'].iloc[-1], df1['p_z'].iloc[-1], color='red', label='End', s=60, zorder=5)

    # Formatting
    ax1.set_title('Simulated Flight Dynamics:\nAscending Square Profile', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Position X [m]', labelpad=10, fontsize=11)
    ax1.set_ylabel('Position Y [m]', labelpad=10, fontsize=11)
    ax1.set_zlabel('Position Z [m]', labelpad=10, fontsize=11)
    ax1.legend(loc='upper right')

    # ---------------------------------------------------------
    # Subplot 2: Upward Spiral Pattern
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(df2['p_x'], df2['p_y'], df2['p_z'], label='Ground Truth Path', color='purple', linewidth=1.5)

    # Mark Start and End
    ax2.scatter(df2['p_x'].iloc[0], df2['p_y'].iloc[0], df2['p_z'].iloc[0], color='green', label='Start', s=60, zorder=5)
    ax2.scatter(df2['p_x'].iloc[-1], df2['p_y'].iloc[-1], df2['p_z'].iloc[-1], color='red', label='End', s=60, zorder=5)

    # Formatting
    ax2.set_title('Simulated Flight Dynamics:\nUpward Spiral Profile', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Position X [m]', labelpad=10, fontsize=11)
    ax2.set_ylabel('Position Y [m]', labelpad=10, fontsize=11)
    ax2.set_zlabel('Position Z [m]', labelpad=10, fontsize=11)
    ax2.legend(loc='upper right')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('mocap_trajectories.png', dpi=300)
    print("Saved side-by-side plot as 'mocap_trajectories.png'")

if __name__ == "__main__":
    plot_synthetic_profiles()