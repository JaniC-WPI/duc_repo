import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_norm(vector):
    """Compute the norm (magnitude) of a vector."""
    return np.linalg.norm(vector)

def process_json_files(folder_path):
    position_norms = []
    joint_displacement_norms = []

    # Iterate through all JSON files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            with open(os.path.join(folder_path, file_name), 'r') as file:
                data = json.load(file)
                
                # Extract position and actual_joint_displacement
                position = data['position']
                actual_joint_displacement = data['actual_joint_displacement']

                # Compute norms
                position_norm = compute_norm(position)
                joint_displacement_norm = compute_norm(actual_joint_displacement)

                # Store norms
                position_norms.append(position_norm)
                joint_displacement_norms.append(joint_displacement_norm)

    return position_norms, joint_displacement_norms

def plot_comparisons(position_norms, joint_displacement_norms):
    # Scatter Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(position_norms, joint_displacement_norms, alpha=0.7)
    plt.xlabel('Custom joint displacement')
    plt.ylabel('Actual joint displacement')
    plt.title('Comparison of Position Norm and Actual Joint Displacement Norm')
    plt.grid(True)
    # plt.show()

    # Histogram Plot for Position Norms
    plt.figure(figsize=(10, 5))
    plt.hist(position_norms, bins=5, alpha=0.7, label='Custom joint displacements')
    plt.hist(joint_displacement_norms, bins=5, alpha=0.7, label='Actual joint displacements')
    plt.xlabel('Norm Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Norms', fontsize=16)    
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_vs_actual.png')
    plt.show()

    # Line Plot to observe trends
    plt.figure(figsize=(10, 5))
    plt.plot(position_norms, label='Position Norms', marker='o')
    plt.plot(joint_displacement_norms, label='Joint Displacement Norms', marker='x')
    plt.xlabel('File Index')
    plt.ylabel('Norm Value')
    plt.title('Trends of Norms Across JSON Files')
    plt.legend()
    plt.grid(True)
    # plt.show()

import os
import json
import numpy as np
import matplotlib.pyplot as plt

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

def plot_boxplots(positions, joint_displacements):
    # Separate the data into components
    joint1_approx = positions[:, 0]
    joint2_approx = positions[:, 1]
    joint3_approx = positions[:, 2]
    
    joint1_actual = joint_displacements[:, 0]
    joint2_actual = joint_displacements[:, 1]
    joint3_actual = joint_displacements[:, 2]

    # Create subplots for each joint
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))
    fig.suptitle('Box Plot Comparison of Approximated and Actual and Joint Displacements')

    # Data for plotting
    data_joint1 = [joint1_approx, joint1_actual]
    data_joint2 = [joint2_approx, joint2_actual]
    data_joint3 = [joint3_approx, joint3_actual]
    
    # Colors for the boxes
    colors = ['blue', 'green']  # Blue for "Position", Green for "Joint Displacement"

    # Plot Joint 1
    bp1 = axes[0].boxplot(data_joint1, patch_artist=True, showfliers=False)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    axes[0].set_title('Joint 1 Comparison')
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(['Approximated', 'Actual'])
    axes[0].set_ylabel('Value')

    axes[0].set_ylim(-0.0015, 0.0015)


    # Plot Joint 2
    bp2 = axes[1].boxplot(data_joint2, patch_artist=True, showfliers=False)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    axes[1].set_title('Joint 2 Comparison')
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(['Approximated', 'Actual'])
    axes[1].set_ylabel('Value')

    # Plot Joint 3
    bp3 = axes[2].boxplot(data_joint3, patch_artist=True, showfliers=False)
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
    axes[2].set_title('Joint 3 Comparison')
    axes[2].set_xticks([1, 2])
    axes[2].set_xticklabels(['Approximated', 'Actual'])
    axes[2].set_ylabel('Value')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust the top space to fit the suptitle
    plt.show()
    
# Folder path to JSON files
folder_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/regression_rearranged_all_double_corrected/'

# Process the JSON files for boxplot
positions, joint_displacements = process_json_files(folder_path)

# Plot the box plot comparison
plot_boxplots(positions, joint_displacements)