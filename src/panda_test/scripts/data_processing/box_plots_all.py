import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def compute_norm(vector):
    """Compute the norm (magnitude) of a vector."""
    return np.linalg.norm(vector)

def process_json_files(folder_path):
    positions = []
    joint_displacements = []

    # Iterate through all JSON files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            with open(os.path.join(folder_path, file_name), 'r') as file:
                data = json.load(file)
                
                # Extract position and actual_joint_displacement
                position = data['position']
                actual_joint_displacement = data['actual_joint_displacement']

                # Store values
                positions.append(position)
                joint_displacements.append(actual_joint_displacement)

    return np.array(positions), np.array(joint_displacements)

def bold_text(text):
    """Helper function to bold text."""
    return f'\\textbf{{{text}}}'

def plot_boxplots_with_external_table(positions, joint_displacements):
    # Separate the data into components
    joint1_approx = positions[:, 0]
    joint2_approx = positions[:, 1]
    joint3_approx = positions[:, 2]
    
    joint1_actual = joint_displacements[:, 0]
    joint2_actual = joint_displacements[:, 1]
    joint3_actual = joint_displacements[:, 2]

    # Create subplots for each joint
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Box Plot Comparison of Approximated and Actual Joint Displacements')

    # Data for plotting
    data_joint1 = [joint1_approx, joint1_actual]
    data_joint2 = [joint2_approx, joint2_actual]
    data_joint3 = [joint3_approx, joint3_actual]
    
    # Colors for the boxes
    colors = ['#FFFF00', '#FF00FF']  # Yellow for "Approximated", Magenta for "Actual"

    # Store statistics for the table
    stats = []

    def set_dynamic_limits(ax, data):
        """Set dynamic y-axis limits to make the boxplots more visible."""
        data_min = np.min(data)
        data_max = np.max(data)
        ax.set_ylim([data_min - 0.1 * np.abs(data_min), data_max + 0.1 * np.abs(data_max)])

    # Plot Joint 1
    bp1 = axes[0].boxplot(data_joint1, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    # Set dynamic limits for joint 1
    set_dynamic_limits(axes[0], data_joint1)
    
    # Calculate mean, std, and median
    means_joint1 = [np.mean(joint1_approx), np.mean(joint1_actual)]
    stds_joint1 = [np.std(joint1_approx), np.std(joint1_actual)]
    median_joint1 = [np.median(joint1_approx), np.median(joint1_actual)]
    stats.append([f'{means_joint1[0]:.8f}', f'{stds_joint1[0]:.8f}', f'{median_joint1[0]}', f'{means_joint1[1]:.8f}', f'{stds_joint1[1]:.8f}', f'{median_joint1[1]}'])
    
    axes[0].set_title('Joint 1 Comparison')
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(['Approximated', 'Actual'])
    axes[0].set_ylabel('Value')

    # Plot Joint 2
    bp2 = axes[1].boxplot(data_joint2, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    # Set dynamic limits for joint 2
    set_dynamic_limits(axes[1], data_joint2)
    
    # Calculate mean, std, and median
    means_joint2 = [np.mean(joint2_approx), np.mean(joint2_actual)]
    stds_joint2 = [np.std(joint2_approx), np.std(joint2_actual)]
    median_joint2 = [np.median(joint2_approx), np.median(joint2_actual)]
    stats.append([f'{means_joint2[0]:.8f}', f'{stds_joint2[0]:.8f}', f'{median_joint2[0]}', f'{means_joint2[1]:.8f}', f'{stds_joint2[1]:.8f}', f'{median_joint2[1]}'])

    axes[1].set_title('Joint 2 Comparison')
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(['Approximated', 'Actual'])

    # Plot Joint 3
    bp3 = axes[2].boxplot(data_joint3, patch_artist=True)
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
    # Set dynamic limits for joint 3
    set_dynamic_limits(axes[2], data_joint3)

    # Calculate mean and std
    means_joint3 = [np.mean(joint3_approx), np.mean(joint3_actual)]
    stds_joint3 = [np.std(joint3_approx), np.std(joint3_actual)]
    median_joint3 = [np.median(joint3_approx), np.median(joint3_actual)]
    stats.append([f'{means_joint3[0]:.8f}', f'{stds_joint3[0]:.8f}', f'{median_joint3[0]}', f'{means_joint3[1]:.8f}', f'{stds_joint3[1]:.8f}', f'{median_joint3[1]}'])

    axes[2].set_title('Joint 3 Comparison')
    axes[2].set_xticks([1, 2])
    axes[2].set_xticklabels(['Approximated', 'Actual'])

    # Create a new figure for the statistics table
    fig_table, ax_table = plt.subplots(figsize=(14, 5))  # Increase the size as needed
    ax_table.axis('tight')
    ax_table.axis('off')
    col_labels = ['Mean Approx', 'Std Dev Approx', 'Median Approx', 'Mean Actual', 'Std Dev Actual', 'Median Actual']
    row_labels = ['Joint 1', 'Joint 2', 'Joint 3']

    bold_font = FontProperties(weight='bold')
    table = ax_table.table(cellText=stats, colLabels=col_labels, rowLabels=row_labels, cellLoc='center', loc='center', fontsize=12)  # Adjust fontsize here


    # for key, cell in table.get_celld().items():
    #     cell.set_fontproperties(bold_font)

    # Increase the font size of the table title if necessary
    table.auto_set_font_size(False)
    table.set_fontsize(14)  # Set table font size
    table.scale(1.2, 1.5)  # Adjust the table size scale (width, height)

    plt.show()

# Folder path to JSON files
folder_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/regression_rearranged_all_corrected/'

# Process the JSON files for boxplot
positions, joint_displacements = process_json_files(folder_path)

# Plot the box plot comparison
plot_boxplots_with_external_table(positions, joint_displacements)