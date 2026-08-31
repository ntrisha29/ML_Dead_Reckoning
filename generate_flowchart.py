import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_horizontal_flowchart():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.axis('off')  # Turn off the axes

    # Define the stages in the pipeline
    stages = [
        "Raw Telemetry\nIngestion",
        "Data Cleansing &\nSynchronization",
        "Feature & Target\nExtraction",
        "Temporal Segmentation\n(Sliding Window)",
        "Tensor Conversion\n& Normalization",
        "Multi-Worker\nBatching",
        "PINN-GRU\nArchitecture"
    ]

    # Styling parameters
    box_width = 2.2
    box_height = 1.2
    y_center = 0.5
    x_start = 0.5
    spacing = 2.8  # Distance between the start of one box and the next

    # Draw boxes and text
    boxes = []
    for i, stage in enumerate(stages):
        x = x_start + i * spacing
        
        # Determine colors (highlighting the final architecture block)
        facecolor = '#f0f4f8' if i < len(stages) - 1 else '#d9e2ec'
        edgecolor = '#2f3e46'
        
        # Create rectangle
        rect = patches.Rectangle(
            (x, y_center - box_height / 2), 
            box_width, 
            box_height, 
            linewidth=1.5, 
            edgecolor=edgecolor, 
            facecolor=facecolor, 
            zorder=2,
            boxstyle="round,pad=0.1" # slight rounding for a polished look
        )
        ax.add_patch(rect)
        boxes.append((x, x + box_width))

        # Add text
        ax.text(
            x + box_width / 2, 
            y_center, 
            stage, 
            horizontalalignment='center', 
            verticalalignment='center', 
            fontsize=11, 
            fontweight='bold' if i == len(stages) - 1 else 'normal',
            color='#102a43',
            zorder=3
        )

    # Draw arrows connecting the boxes
    for i in range(len(boxes) - 1):
        x_tail = boxes[i][1]  # Right edge of current box
        x_head = boxes[i+1][0]  # Left edge of next box
        
        # Create an arrow
        arrow = patches.FancyArrowPatch(
            (x_tail, y_center), 
            (x_head, y_center),
            arrowstyle='-|>',
            mutation_scale=15,
            linewidth=1.5,
            color='#486581',
            zorder=1
        )
        ax.add_patch(arrow)

    # Add branch details below the Extraction box (Index 2)
    extraction_x_center = x_start + 2 * spacing + box_width / 2
    ax.text(
        extraction_x_center, 
        y_center - box_height / 2 - 0.2, 
        "Branch A: 6-DoF IMU Matrices\nBranch B: Kinematic Finite Differencing", 
        horizontalalignment='center', 
        verticalalignment='top', 
        fontsize=9, 
        fontstyle='italic',
        color='#627d98'
    )

    # Add title
    plt.title(
        "Data Pre-Processing and Signal Conditioning Pipeline", 
        fontsize=16, 
        fontweight='bold', 
        color='#102a43', 
        pad=20
    )

    # Adjust layout and save the image
    plt.tight_layout()
    plt.savefig('preprocessing_flowchart_horizontal.png', dpi=300, bbox_inches='tight')
    print("Horizontal block diagram successfully saved as 'preprocessing_flowchart_horizontal.png'")

if __name__ == "__main__":
    draw_horizontal_flowchart()