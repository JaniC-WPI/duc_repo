import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import pickle


def load_graph(file_path):
    """
    Load a graph object from a pickle file.

    Args:
    - file_path (str): Path to the pickle file.

    Returns:
    - graph: Loaded graph object.
    """
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    return graph

def plot_joint_rm_3d(roadmap):
    joint_angles = [roadmap.nodes[node]['joint_angles'] for node in roadmap.nodes]
    j1_angles = [angles[0] for angles in joint_angles]
    j2_angles = [angles[1] for angles in joint_angles]
    j3_angles = [angles[2] for angles in joint_angles]

    # Create a DataFrame for Seaborn visualizations
    joint_angles_df = pd.DataFrame({
        'Joint 1': j1_angles,
        'Joint 2': j2_angles,
        'Joint 3': j3_angles
    })
    
    # sns.set_theme(style='white')

    # 3D Scatter Plot using Matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    

    # Plotting all configurations as points in 3D space
    scatter = ax.scatter(j1_angles, j2_angles, j3_angles, c='#2E23C8', marker='o', s=10)

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14, pad=10)

    # Set bold font for tick labels
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax.get_zticklabels():
        label.set_fontweight('bold')

    # Setting labels for axes
    # ax.set_xlabel('Joint 1', fontsize=10, fontweight='bold')
    # ax.set_ylabel('Joint 2', fontsize=10, fontweight='bold')
    # ax.set_zlabel('Joint 3', fontsize=10, fontweight='bold')
    # ax.set_title('3D Plot of Joint Angles in the Entire Dataset')

    # Display the plot
    plt.show()

def plot_joint_angle_histograms(roadmap, label, output_path):
    # Extract joint angles from the roadmap
    joint_angles = [roadmap.nodes[node]['joint_angles'] for node in roadmap.nodes]
    joint_1 = [angles[0] for angles in joint_angles]
    joint_2 = [angles[1] for angles in joint_angles]
    joint_3 = [angles[2] for angles in joint_angles]

    # Create a DataFrame for Seaborn visualizations
    joint_angles_df = pd.DataFrame({
        'Joint 1': joint_1,
        'Joint 2': joint_2,
        'Joint 3': joint_3
    })

    # Calculate the common range and set a consistent bin width
    min_value = min(joint_angles_df.min())
    max_value = max(joint_angles_df.max())
    bin_width = 0.25  # Set your desired bin width
    bins = np.arange(min_value, max_value + bin_width, bin_width)  # Create bins with the same width

    # Plot individual histograms for each joint angle with the same bin width
    plt.figure(figsize=(20, 10))
    for i, joint in enumerate(joint_angles_df.columns, 1):
        plt.subplot(1, 3, i)
        sns.histplot(joint_angles_df[joint], bins=bins, kde=False, color='#5D3A9B', alpha = 0.9)
        # plt.title(f'Distribution of {joint}', fontsize=14, fontweight='bold')
        # plt.xlabel(joint, fontsize=12, fontweight='bold')
        # plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        
        plt.xlabel('')
        plt.ylabel('')
        plt.xlim(-3, 3)
        plt.ylim(0, 5301)
        plt.tick_params(axis='y', pad=5)
        plt.xticks(fontsize=32, weight='bold')
        # plt.yticks(fontsize=32, weight='bold')
        if i == 1:
            # plt.yticks(fontsize=32, weight='bold')
            plt.yticks(np.arange(0, 5301, 500), fontsize=32, weight='bold')
        else:
            plt.yticks([])
        


    # plt.suptitle(f'Joint Angle Distributions for {label}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    plt.close()

folder_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged_og/'

custom_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_roadmap_angle_fresh_432.pkl'
gt_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_roadmap_angle_fresh_432.pkl'

gt_graph = load_graph(gt_graph_path)

# plot_joint_rm_3d(gt_graph)


# Initialize lists to hold joint angles
j1_angles = []
j2_angles = []
j3_angles = []

# Iterate over each file in the folder
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith('_joint_angles.json'):  # Ensures we only read .json files
        # Check if the file index is greater than or equal to 010000
        file_index = int(filename.split('_')[0])
        # if file_index >= 10000:
        file_path = os.path.join(folder_path, filename)
        
        # Open and read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
            joint_angles = data['joint_angles']
            
            # Append each angle to its respective list
            j1_angles.append(joint_angles[0])
            j2_angles.append(joint_angles[1])
            j3_angles.append(joint_angles[2])

# sns.set_theme(style='white')

# Create a DataFrame for Seaborn visualizations
joint_angles_df = pd.DataFrame({
    'Joint 1': j1_angles,
    'Joint 2': j2_angles,
    'Joint 3': j3_angles
})

# 3D Scatter Plot using Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting all configurations as points in 3D space
scatter = ax.scatter(j1_angles, j2_angles, j3_angles, c='#517119', marker='o', s=5)

ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='z', labelsize=14, pad=10)

# Set bold font for tick labels
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
for label in ax.get_zticklabels():
    label.set_fontweight('bold')

# Setting labels for axes
# ax.set_xlabel('Joint 1', fontsize=10, fontweight='bold')
# ax.set_ylabel('Joint 2', fontsize=10, fontweight='bold')
# ax.set_zlabel('Joint 3', fontsize=10, fontweight='bold')
# ax.set_title('3D Plot of Joint Angles in the Entire Dataset')

# Display the plot
# plt.show()



# 2D Scatter Plot using Matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(joint_angles_df['Joint 1'], joint_angles_df['Joint 3'], c='b', marker='o', s=10)

# Increase the size of the tick labels and set them bold
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

# Setting labels and title with bold font
plt.xlabel('Joint 1', fontsize=14, fontweight='bold')
plt.ylabel('Joint 2', fontsize=14, fontweight='bold')
plt.title('2D Plot of Joint 1 vs Joint 2', fontsize=16, fontweight='bold')

# Display the plot
plt.grid(True)
# plt.show()

# # Pairplot for Pairwise Joint Distribution using Seaborn
# sns.pairplot(joint_angles_df, diag_kind='kde', plot_kws={'alpha': 0.6})
# plt.suptitle('Pairwise Joint Angle Distributions', y=1.02, fontsize=16, fontweight='bold')
# plt.show()


# KDE plots for individual joint angle distributions
# plt.figure(figsize=(20, 10))
# for i, joint in enumerate(joint_angles_df.columns, 1):
#     plt.subplot(1, 3, i)
#     sns.kdeplot(joint_angles_df[joint], shade=True, color='#1f77b4')
#     plt.title(f'Distribution of {joint}', fontsize=14, fontweight='bold')
#     plt.xlabel(joint, fontsize=12, fontweight='bold')
#     plt.ylabel('Density', fontsize=12, fontweight='bold')

# plt.tight_layout()
# plt.show()

# Calculate the common range and set a consistent bin width
min_value = min(joint_angles_df.min())
max_value = max(joint_angles_df.max())
bin_width = 0.25  # Set your desired bin width
bins = np.arange(min_value, max_value + bin_width, bin_width)

plt.figure(figsize=(20, 10))
# sns.set_theme(style='white')
for i, joint in enumerate(joint_angles_df.columns, 1):
    plt.subplot(1, 3, i)    
    sns.histplot(joint_angles_df[joint], bins=bins, kde=False, color='#517119', edgecolor="black",
            alpha=0.9)
    # plt.title(f'Distribution of {joint}', fontsize=14, fontweight='bold')
    # plt.xlabel(joint, fontsize=12, fontweight='bold')
    # plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('')
    # plt.grid(False)
    plt.xlim(-3, 3)
    # plt.ylim(-1, 5300)
    plt.xticks(fontsize=32, weight='bold')
    # plt.yticks(fontsize=18, weight='bold')
    if i == 1:
            # plt.yticks(fontsize=30, weight='bold')
            plt.yticks(np.arange(0, 5301, 500), fontsize=32, weight='bold')        
    else:
        plt.yticks([])

plt.tight_layout()
plt.show()
plt.close()

plot_joint_angle_histograms(gt_graph, 'Custom Distance', 'custom_roadmap_joint_angles.png')

