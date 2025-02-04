import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Function to calculate Euclidean distances between consecutive joint configurations
def calculate_kp_jt_distances(file_path):
    df = pd.read_csv(file_path)

    joint_columns = df[['Joint 1', 'Joint 2', 'Joint 3']].to_numpy()

    # Compute distances for each joint
    j1_distances = np.sqrt(np.diff(joint_columns[:, 0]) ** 2)  # Joint 1
    j2_distances = np.sqrt(np.diff(joint_columns[:, 1]) ** 2)  # Joint 2
    j3_distances = np.sqrt(np.diff(joint_columns[:, 2]) ** 2)  # Joint 
    j1_dist_norm = (j1_distances - np.min(j1_distances))/(np.max(j1_distances) - np.min(j1_distances))
    j2_dist_norm = (j2_distances - np.min(j2_distances))/(np.max(j2_distances) - np.min(j2_distances))
    j3_dist_norm = (j3_distances - np.min(j3_distances))/(np.max(j3_distances) - np.min(j3_distances))

    x_keypoints = df.iloc[:, 1:18:2].to_numpy()  # Odd columns: 1, 3, ..., 17
    y_keypoints = df.iloc[:, 2:19:2].to_numpy()  # Even columns: 2, 4, ..., 18

    all_keypoints = np.concatenate([x_keypoints, y_keypoints], axis=1)

    kp_diff = np.diff(all_keypoints, axis=0)
    kp_distances = np.linalg.norm(kp_diff, axis=1)

    kp_dist_norm = (kp_distances - np.min(kp_distances)) / (np.max(kp_distances) - np.min(kp_distances))
    
    return j1_dist_norm, j2_dist_norm, j3_dist_norm, kp_dist_norm

def plot_joint_and_keypoint_distances(file_paths, labels, colors):

    joint_columns = ['Joint 1', 'Joint 2', 'Joint 3']
    fig, axes = plt.subplots(4, 1, figsize=(18, 18))  # Adjusted to 4 subplots

    max_value = 0

    # Determine the global maximum value across all files and metrics
    for file_path in file_paths:
        j1_dist_norm, j2_dist_norm, j3_dist_norm, kp_distances = calculate_kp_jt_distances(file_path)

        max_jt_value = max(np.max(distances) for distances in [j1_dist_norm, j2_dist_norm, j3_dist_norm])  # Max across all joints
        max_kp_value = np.max(kp_distances)  # Max for keypoints
        max_value = max(max_value, max_jt_value, max_kp_value)  # Global max

    # Plot Keypoint Distances in the First Subplot
    ax_kp = axes[0]
    for file_path, label, color in zip(file_paths, labels, colors):
        _,_,_, kp_distances = calculate_kp_jt_distances(file_path)
        sns.lineplot(
            x=range(1, len(kp_distances) + 1),
            y=kp_distances,
            marker='o',
            markersize=20,
            linewidth=4,
            label=label,
            ax=ax_kp,
            color=color
        )

    # ax_kp.set_title('Keypoint Distances', fontsize=22, fontweight='bold')
    ax_kp.set_ylim(-0.1, max_value + 0.1)
    ax_kp.tick_params(axis='both', which='major', labelsize=14)

    for label in ax_kp.get_xticklabels():
        ax_kp.set_xticklabels([])

    for label in ax_kp.get_yticklabels():
        label.set_fontsize(20)
        label.set_fontweight('bold')

    ax_kp.legend(prop={'size': 17, 'weight': 'bold'}, loc='upper left', bbox_to_anchor=(-0.0009, 1.035))
    ax_kp.grid(False)

    # Plot Joint Distances in the Remaining Subplots
    for joint_idx, joint in enumerate(joint_columns):
        ax = axes[joint_idx + 1]  # Remaining subplots start from index 1
        for file_path, label, color in zip(file_paths, labels, colors):
            j1_dist_norm, j2_dist_norm, j3_dist_norm, _ = calculate_kp_jt_distances(file_path)
            jt_distances = [j1_dist_norm, j2_dist_norm, j3_dist_norm]
            sns.lineplot(
                x=range(1, len(jt_distances[joint_idx]) + 1),
                y=jt_distances[joint_idx],  # Extract distances for the specific joint
                marker='o',
                markersize=20,
                linewidth=4,
                # label=f'{label}' if joint_idx == 0 else '',
                ax=ax,
                color=color
            )

        # ax.set_title(f'Distances Between Configurations for {joint}', fontsize=22, fontweight='bold')
        ax.set_ylim(-0.1, max_value + 0.1)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xticks(range(1, len(jt_distances[joint_idx]) + 1, max(1, len(jt_distances[joint_idx]) // 10)))  # Whole number ticks

        for label in ax.get_xticklabels():
            label.set_fontsize(20)
            label.set_fontweight('bold')

        for label in ax.get_yticklabels():
            label.set_fontsize(20)
            label.set_fontweight('bold')

        # Only set x-tick labels for the last subplot
        if joint_idx == len(joint_columns) - 1:
            ax.set_xlabel('Configuration Pair Index', fontsize=22, fontweight='bold', labelpad=20)
            ax.set_ylabel('distances', fontsize=22, fontweight='bold', labelpad=20)
        else:
            ax.set_xticklabels([])  # Remove x-tick labels for the first two subplots

        # Add legend only to the first subplot
        # if joint_idx == 0:
        #     ax.legend('')

        ax.grid(False)

    plt.tight_layout()
    plt.show()


# Function to calculate joint distances for each joint between configurations
# def calculate_joint_distances(file_path):
#     print(file_path)
#     df = pd.read_csv(file_path)
#     joint_columns = ['Joint 1', 'Joint 2', 'Joint 3']
#     joint_distances = {joint: [] for joint in joint_columns}
    
#     for joint in joint_columns:
#         for i in range(len(df) - 1):
#             distance = abs(df[joint].iloc[i + 1] - df[joint].iloc[i])
#             joint_distances[joint].append(distance)
    
#     return joint_distances

# # Function to plot joint distances for a single CSV file using Seaborn
# def plot_joint_distances(file_path, label, color):    
#     joint_distances = calculate_joint_distances(file_path)
#     # sns.set_theme(style="whitegrid")
#     plt.figure(figsize=(14, 7))
    
#     for joint, distances in joint_distances.items():
#         sns.lineplot(x=range(1, len(distances) + 1), y=distances, marker='o', markersize=10, linewidth=3, label=f'{label} - {joint}', color=color)

#     plt.legend([], [], frameon=False)
#     plt.yticks(fontsize=24, weight='bold')
    
#     # plt.title('Joint Distances Between Configurations')
#     # plt.xlabel('Configuration Pair Index')
#     # plt.ylabel('Distance')
#     plt.legend(prop={'size': 24, 'weight': 'bold'})
#     plt.show()

# # Function to plot distances for each joint separately for all three file paths using Seaborn
# def plot_joint_distances_for_all_files(file_paths, labels, colors):
#     joint_columns = ['Joint 1', 'Joint 2', 'Joint 3']
#     fig, axes = plt.subplots(3, 1, figsize=(24, 15))
#     # sns.set_theme(style="whitegrid")
#     max_joint_value = 0
#     for file_path in file_paths:
#         joint_distances, kp_distances = calculate_kp_jt_distances(file_path)
#         max_joint_value = max(max_joint_value, max(max(distances) for distances in joint_distances.values()))
    
#     for joint_idx, joint in enumerate(joint_columns):
#         ax = axes[joint_idx]
#         for file_path, label, color in zip(file_paths, labels, colors):
#             print(file_path)
#             joint_distances, kp_distances = calculate_kp_jt_distances(file_path)
#             sns.lineplot(
#                 x=range(1, len(joint_distances[joint]) + 1),
#                 y=joint_distances[joint],
#                 marker='o',
#                 markersize=20,
#                 linewidth=4,
#                 label=f'{label}' if joint_idx == 0 else '',
#                 ax=ax,
#                 color=color
#             )

#         # Remove the legend for each subplot
#         if ax.get_legend() is '':
#             ax.get_legend().remove()

#         ax.tick_params(axis='both', which='major', labelsize=14)
#         # Customize x-axis ticks to show whole numbers
#         max_ticks = len(joint_distances[joint])
#         step = 2  # Customize the step size for the tick interval if needed
#         ax.set_xticks(range(1, max_ticks + 1, step))
#         ax.set_xticklabels([str(i) for i in range(1, max_ticks + 1, step)], fontsize=14, weight='bold')

#         ax.set_ylim(-0.07, max_joint_value+0.1)

#         # Set tick label fonts to bold manually
#         for label in ax.get_xticklabels():
#             label.set_fontsize(20)
#             label.set_fontweight('bold')

#         for label in ax.get_yticklabels():
#             label.set_fontsize(20)
#             label.set_fontweight('bold')

#         # Only set x-tick labels for the last subplot
#         if joint_idx == len(joint_columns) - 1:
#             ax.set_xticks(range(1, max_ticks + 1, step))
#             ax.set_xticklabels([str(i) for i in range(1, max_ticks + 1, step)], fontsize=22, weight='bold')
#         else:
#             ax.set_xticklabels([])  # Remove x-tick labels for the first two subplots
        
#         # ax.set_title(f'Distances Between Configurations for {joint}')
#         # ax.set_xlabel('Configuration Pair Index')
#         # ax.set_ylabel('Distance')
#         # ax.legend(prop={'size': 20, 'weight': 'bold'})
#         # Add the legend only to the first subplot
#         if joint_idx == 0:
#             ax.legend(prop={'size': 20, 'weight': 'bold'})
        

#     plt.grid(False)
#     plt.tight_layout()
#     plt.show()

# def euclidean_distance(row1, row2):
#     """
#     Calculates the Euclidean distance between two rows of keypoints.
    
#     Args:
#     - row1, row2: Arrays or lists containing keypoint x and y coordinates for two configurations.
    
#     Returns:
#     - Euclidean distance between the two configurations.
#     """
#     dist = 0
#     for kp in range(0, len(row1), 2):  # Iterate over keypoints in pairs (x, y)
#         dist += (row2[kp] - row1[kp])**2 + (row2[kp+1] - row1[kp+1])**2
#     return np.sqrt(dist)

# def calculate_distances(file_path):
#     """
#     Reads a CSV file containing keypoint data and calculates Euclidean distances between consecutive configurations.
    
#     Args:
#     - file_path: Path to the CSV file.
    
#     Returns:
#     - distances: List of Euclidean distances between consecutive keypoint configurations.
#     """
#     df = pd.read_csv(file_path)

#     # Select only the first 18 columns after 'Config' (representing the keypoint x, y pairs)
#     keypoint_columns = df.iloc[:, 1:19].values  # We limit to exactly 18 columns (9 keypoint pairs)

#     distances = []
#     for i in range(len(df) - 1):
#         distances.append(euclidean_distance(keypoint_columns[i], keypoint_columns[i+1]))

#     return distances

# def plot_distances(file_path1, file_path2, file_path3):
#     # sns.set_theme(style="whitegrid")
#     plt.figure(figsize=(20, 10))

#     # Calculate distances for each file
#     distances1 = calculate_distances(file_path1)
#     distances2 = calculate_distances(file_path2)
#     distances3 = calculate_distances(file_path3)

#     # Plot each set of distances using Seaborn
#     sns.lineplot(x=range(1, len(distances1) + 1), y=distances1, marker='o', markersize=20,
#                 linewidth=4, label='Ground Truth', color='#40B0A6')
#     sns.lineplot(x=range(1, len(distances2) + 1), y=distances2, marker='o', markersize=20,
#                 linewidth=4, label='Learned', color='#5D3A9B')
#     sns.lineplot(x=range(1, len(distances3) + 1), y=distances3, marker='o', markersize=20,
#                 linewidth=4, label='Image Space', color='#D41159')
    
#      # Customize the x-ticks to show whole numbers
#     max_xticks = max(len(distances1), len(distances2), len(distances3))
#     step = 2  # Adjust the step size as needed
#     plt.xticks(range(1, max_xticks + 1, step), fontsize=24, weight='bold')

#     # Set y-ticks and other properties
#     plt.yticks(fontsize=24, weight='bold')

    # Set plot labels and legend
    # plt.title("Comparison of Euclidean Distances Between Keypoint Configurations", fontsize=20, weight='bold')
    # plt.xlabel("Configuration Pair Index", fontsize=18, weight='bold')
    # plt.ylabel("Euclidean Distance", fontsize=18, weight='bold')
    # plt.legend(prop={'size': 18, 'weight': 'bold'})
    # plt.legend().remove()
    # plt.legend(prop={'size': 24, 'weight': 'bold'})


    # # Show the plot
    # plt.grid(False)
    # plt.show()

def main():
    # Paths to the CSV files
    file_path1 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/with_obs/nn_25_astar_custom_old/1/save_distances.csv'
    file_path2 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/with_obs/nn_25_astar_custom_old/1/save_distances.csv'
    file_path3 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/with_obs/1/save_distances.csv'

    labels = ['Ground Truth', 'Learned', 'Image Space']
    colors = ['#40B0A6', '#5D3A9B', '#D41159'] 
    
    # Plot joint distances between configurations for each file
    # plot_joint_distances(file_path1, labels[0], colors[0])
    # plot_joint_distances(file_path2, labels[1], colors[1])
    # plot_joint_distances(file_path3, labels[2], colors[2])

    # Plot joint distances for all three file paths in subplots
    plot_joint_and_keypoint_distances([file_path1, file_path2, file_path3], labels, colors)

    # Plot keypoints from all three CSV files on the same 2D scatter plot

    fig, ax = plt.subplots(figsize=(20, 10))

    # plot_euclidean_joint_distances_combined([file_path3, file_path1, file_path2], labels, colors)

    # plot_distances(file_path1, file_path2, file_path3) 


if __name__ == "__main__":
    main()
