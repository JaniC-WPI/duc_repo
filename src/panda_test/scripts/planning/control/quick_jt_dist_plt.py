# import numpy as np
# import math
# from scipy.spatial.transform import Rotation as R
# from Robot import RobotTest, PandaReal2D
# import matplotlib.pyplot as plt
# import pandas as pd
# import cv2
# import os

# jt_file_paths_template = [
#     '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/with_obs/nn_25_astar_custom_old/{}/save_distances.csv',
#     '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/with_obs/nn_25_astar_custom_old/{}/save_distances.csv',
#     '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/with_obs/{}/save_distances.csv'                                                             
# ]

# # Output directory for saving plots
# output_dir = "/media/jc-merlab/Crucial X9/paper_data/distance_plots"
# os.makedirs(output_dir, exist_ok=True)

# exp_no = [1,2,3,4,6,8,9,10,13,14,15,16,17,18,19,20] # exp with no obs
# # exp_no = [1, 2, 8, 9, 10, 14, 17, 18, 19, 20] # exps for with obs
# # exp_no = [1]

# labels = ["Ground Truth", "Learned", "Image Space"]


# # Loop through each label and corresponding file path
# for label, path_template in zip(labels, jt_file_paths_template):
#     for exp in exp_no:
#         # Format the file path for the current experiment
#         file_path = path_template.format(exp)
        
#         # Check if the file exists
#         if not os.path.exists(file_path):
#             print(f"File not found: {file_path}")
#             continue

#         # Read the CSV file
#         df = pd.read_csv(file_path)

#         # Check if the column exists
#         if "Distance to next Joint Angles" not in df.columns:
#             print(f"'Distance to next Joint Angles' column not found in {file_path}")
#             continue

#         # Extract the "Distance to Next Joint Angles" column
#         joint_angle_distances = df['Distance to next Joint Angles']

#         # Create the plot
#         plt.figure(figsize=(10, 6))
#         plt.plot(joint_angle_distances, label=f"Exp {exp}")
#         plt.xlabel('Configuration Index', fontsize=12)
#         plt.ylabel('Distance to Next Joint Angle', fontsize=12)
#         plt.title(f'Distance to Next Joint Angles for {label} (Exp {exp})', fontsize=14)
#         plt.legend(fontsize=10)
#         plt.xlim(0, 25)
#         plt.ylim(-1, 1.5)
#         plt.grid(True)
#         plt.tight_layout()

#         # Generate the output file name
#         label_short = label.replace(" ", "_").lower()
#         file_name = f"with_obs_{label_short}_exp_{exp:02d}.png"
#         output_path = os.path.join(output_dir, file_name)

#         # Save the plot
#         plt.savefig(output_path)
#         plt.close()

#         print(f"Saved plot: {output_path}")

# import numpy as np
# import math
# from scipy.spatial.transform import Rotation as R
# from Robot import RobotTest, PandaReal2D
# import matplotlib.pyplot as plt
# import pandas as pd
# import cv2
# import os

# jt_file_paths_template = [
#     '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/with_obs/nn_25_astar_custom_old/{}/save_distances.csv',
#     '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/with_obs/nn_25_astar_custom_old/{}/save_distances.csv',
#     '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/with_obs/{}/save_distances.csv'                                                             
# ]

# # Output directory for saving plots
# output_dir = "/media/jc-merlab/Crucial X9/paper_data/distance_plots"
# os.makedirs(output_dir, exist_ok=True)

# exp_no = [1,2,3,4,6,8,9,10,13,14,15,16,17,18,19,20] # exp with no obs
# # exp_no = [1, 2, 8, 9, 10, 14, 17, 18, 19, 20] # exps for with obs
# # exp_no = [1]

# labels = ["Ground Truth", "Learned", "Image Space"]
# colors = ['#40B0A6', '#5D3A9B', '#D41159']  # Assign unique colors for each type

# # Loop through each experiment
# # Loop through each experiment
# for exp in exp_no:
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#     fig.suptitle(f'Joint Angles for Exp {exp}', fontsize=16)

#     # Loop through each label and corresponding file path
#     for label, path_template, color in zip(labels, jt_file_paths_template, colors):
#         # Format the file path for the current experiment
#         file_path = path_template.format(exp)
        
#         # Check if the file exists
#         if not os.path.exists(file_path):
#             print(f"File not found: {file_path}")
#             continue

#         # Read the CSV file
#         df = pd.read_csv(file_path)

#         # Check if the required columns exist
#         joint_columns = ["Joint 1", "Joint 2", "Joint 3"]
#         if not all(col in df.columns for col in joint_columns):
#             print(f"Required joint columns not found in {file_path}")
#             continue

#         # Plot joint angles for each joint
#         for joint_idx, joint_col in enumerate(joint_columns):
#             joint_angles = df[joint_col]

#             # Plot on the corresponding subplot
#             axes[joint_idx].plot(joint_angles, label=label, color=color, marker='o', markersize=6, linestyle='-')
#             axes[joint_idx].set_title(f'Joint {joint_idx + 1}', fontsize=14)
#             axes[joint_idx].set_xlabel('Configuration Index', fontsize=12)
#             axes[joint_idx].set_ylabel('Joint Angle (rad)', fontsize=12)
#             axes[joint_idx].grid(True)

#     # Add legends to the subplots
#     for ax in axes:
#         ax.legend(fontsize=10)

#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title

#     # Generate the output file name
#     file_name = f"with_obs_exp_{exp:02d}.png"
#     output_path = os.path.join(output_dir, file_name)
#     plt.show()
#     # Save the plot
#     plt.savefig(output_path)
#     plt.close()

#     print(f"Saved plot: {output_path}")

import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from Robot import RobotTest, PandaReal2D
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os

jt_file_paths_template = [
    # '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/with_obs/nn_25_astar_custom_old/{}/save_distances.csv',
    # '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/with_obs/nn_25_astar_custom_old/{}/save_distances.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/with_obs/{}/save_distances.csv'                                                             
]

# Output directory for saving plots
output_dir = "/media/jc-merlab/Crucial X9/paper_data/actual_distance_plots"
os.makedirs(output_dir, exist_ok=True)

# exp_no = [1,2,3,4,6,8,9,10,13,14,15,16,17,18,19,20] # exp with no obs

# exp_no = [1, 2, 8, 9, 10, 14, 17, 18, 19, 20] # exps for with obs

exp_no = [1]


labels = ["Ground Truth", "Learned", "Image Space"]
# labels = ["Ground Truth", "Learned", "Image Space"]
# colors = ['#40B0A6', '#5D3A9B', '#D41159']  # Assign unique colors for each type
colors = ['#D41159']  # Assign unique colors for each type

# markers = ['o', 's', 'D']  # Assign unique markers for each type
marker_sizes = [10, 7, 4]

# Loop through each experiment
for exp in exp_no:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f'Joint Distances for Exp {exp}', fontsize=16)

    # Loop through each label and corresponding file path
    for label, path_template, color, marker_size in zip(labels, jt_file_paths_template, colors, marker_sizes):
        # Format the file path for the current experiment
        file_path = path_template.format(exp)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check if the required columns exist
        joint_columns = ["Joint 1", "Joint 2", "Joint 3"]
        if not all(col in df.columns for col in joint_columns):
            print(f"Required joint columns not found in {file_path}")
            continue

        # Calculate distances between consecutive configurations for each joint
        for joint_idx, joint_col in enumerate(joint_columns):
            joint_angles = df[joint_col].to_numpy()
            # joint_distances = np.abs(np.diff(joint_angles))
            joint_distances = np.diff(joint_angles)

            print(joint_distances)

            # Plot on the corresponding subplot
            axes[joint_idx].plot(joint_distances, label=label, color=color, marker='o', markersize=marker_size, linestyle='-')
            axes[joint_idx].set_title(f'Joint {joint_idx + 1}', fontsize=14)
            axes[joint_idx].set_xlabel('Configuration Index', fontsize=12)
            axes[joint_idx].set_ylabel('Joint Distance (rad)', fontsize=12)
            axes[joint_idx].set_xlim(-1, 25)
            axes[joint_idx].set_ylim(-1, 1.1)
            axes[joint_idx].grid(True)

    # Add legends to the subplots
    for ax in axes:
        ax.legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title

    # Generate the output file name
    file_name = f"joint_distances_with_obs_exp_{exp:02d}.png"
    output_path = os.path.join(output_dir, file_name)

    # Save the plot
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot: {output_path}")

import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from Robot import RobotTest, PandaReal2D
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os

jt_file_paths_template = [
    # '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/with_obs/nn_25_astar_custom_old/{}/save_distances.csv',
    # '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/with_obs/nn_25_astar_custom_old/{}/save_distances.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/with_obs/{}/save_distances.csv'                                                             
]

# Output directory for saving plots
output_dir = "/media/jc-merlab/Crucial X9/paper_data/normalized_distance_plots"
os.makedirs(output_dir, exist_ok=True)

# exp_no = [1,2,3,4,6,8,9,10,13,14,15,16,17,18,19,20] # exp with no obs

exp_no = [1, 2, 8, 9, 10, 14, 17, 18, 19, 20] # exps for with obs

# exp_no = [8]

labels = ["Ground Truth", "Learned", "Image Space"]
colors = ['#40B0A6', '#5D3A9B', '#D41159']  # Assign unique colors for each type
markers = [10, 7, 4]  # Assign unique markers for each type

# Loop through each experiment
for exp in exp_no:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f'Normalized Joint Distances for Exp {exp}', fontsize=16)

    # Loop through each label and corresponding file path
    for label, path_template, color, marker in zip(labels, jt_file_paths_template, colors, markers):
        # Format the file path for the current experiment
        file_path = path_template.format(exp)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check if the required columns exist
        joint_columns = ["Joint 1", "Joint 2", "Joint 3"]
        if not all(col in df.columns for col in joint_columns):
            print(f"Required joint columns not found in {file_path}")
            continue

        # Calculate distances between consecutive configurations for each joint
        for joint_idx, joint_col in enumerate(joint_columns):
            joint_angles = df[joint_col].to_numpy()
            # joint_distances = np.abs(np.diff(joint_angles))
            joint_distances = np.diff(joint_angles)

            # Normalize the joint distances
            min_distance = np.min(joint_distances)
            max_distance = np.max(joint_distances)
            normalized_distances = 2*(joint_distances - min_distance) / (max_distance - min_distance) - 1

            # Plot on the corresponding subplot
            axes[joint_idx].plot(normalized_distances, label=label, color=color, marker='o', markersize=marker, linestyle='-')
            axes[joint_idx].set_title(f'Joint {joint_idx + 1}', fontsize=14)
            axes[joint_idx].set_xlabel('Configuration Index', fontsize=12)
            axes[joint_idx].set_ylabel('Normalized Joint Distance', fontsize=12)
            axes[joint_idx].grid(True)

    # Add legends to the subplots
    for ax in axes:
        ax.legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title

    # Generate the output file name
    file_name = f"normalized_joint_distances_with_obs_exp_{exp:02d}.png"
    output_path = os.path.join(output_dir, file_name)

    # Save the plot
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot: {output_path}")



# import numpy as np

# # List of configurations (keypoints in image space)
# configurations = [
#     [[250, 442], [252, 311], [211, 273], [169, 235], [188, 212], [215, 148], [244, 81], [242, 50], [280, 47]],
#     [[250, 442], [252, 311], [215, 266], [178, 221], [201, 201], [249, 145], [297, 89], [301, 57], [341, 63]],
#     [[250, 442], [252, 311], [231, 256], [209, 202], [238, 190], [310, 175], [383, 160], [407, 137], [437, 167]],
#     [[250, 442], [252, 311], [249, 252], [245, 193], [276, 190], [349, 172], [421, 154], [447, 132], [475, 163]],
#     [[250, 442], [252, 311], [267, 253], [282, 196], [312, 203], [387, 189], [461, 176], [486, 153], [515, 184]],
#     [[250, 442], [252, 311], [267, 253], [282, 196], [312, 203], [386, 188], [461, 174], [485, 151], [515, 181]],
#     [[250, 442], [252, 311], [275, 255], [294, 200], [322, 209], [394, 194], [468, 181], [494, 158], [522, 187]],
# ]

# # Total number of interpolation points
# total_points = 1000

# # Step 1: Compute distances between consecutive configurations
# def compute_distance(config1, config2):
#     """Compute the Euclidean distance between two configurations."""
#     return sum(np.linalg.norm(np.array(kp1) - np.array(kp2)) for kp1, kp2 in zip(config1, config2))

# distances = [compute_distance(configurations[i], configurations[i + 1]) for i in range(len(configurations) - 1)]
# total_distance = sum(distances)

# # Step 2: Compute relative weights
# weights = [dist / total_distance for dist in distances]

# # Step 3: Distribute interpolation points
# interpolation_points = [int(total_points * weight) for weight in weights]

# # Step 4: Adjust for rounding to ensure the total number of points matches
# interpolation_points[-1] += total_points - sum(interpolation_points)  # Adjust the last segment

# # Step 5: Perform interpolation
# interpolated_path = []
# for i, points in enumerate(interpolation_points):
#     start_config = np.array(configurations[i])
#     end_config = np.array(configurations[i + 1])
#     for t in np.linspace(0, 1, points, endpoint=False):  # Interpolate for the given number of points
#         interpolated_config = (1 - t) * start_config + t * end_config
#         interpolated_path.append(interpolated_config)

# # Include the final goal configuration
# interpolated_path.append(np.array(configurations[-1]))

# # Result
# print(f"Total_distance: {total_distance}")
# print(f"Distances: {distances}")
# print(f"weights: {weights}")
# print(f"Interpolation points: {interpolation_points}")
# print(f"Total interpolated points: {len(interpolated_path)}")
    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# File paths template
jt_file_paths_template = [
    # '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/with_obs/nn_25_astar_custom_old/{}/save_distances.csv',
    # '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/with_obs/nn_25_astar_custom_old/{}/save_distances.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/with_obs/{}/save_distances.csv'
]

# Experiment numbers
# exp_no = [1, 2, 8, 9, 10, 14, 17, 18, 19, 20]  # Experiments with observations
exp_no = [1]

# Labels and colors for plots
labels = ["Image Space"]
colors = [['#40B0A6', '#5D3A9B', '#D41159']]  # Unique colors for each type

# Loop through each experiment
# Loop through each experiment
for exp in exp_no:
    plt.figure(figsize=(12, 8))
    # plt.title(f'Joint Distances for Exp {exp}', fontsize=16)
    # plt.xlabel('Configuration Index', fontsize=12)
    # plt.ylabel('Joint Distance (rad)', fontsize=12)
    # plt.grid(True)

    # Loop through each label and corresponding file path
    for label, path_template, color_set in zip(labels, jt_file_paths_template, colors):
        # Format the file path for the current experiment
        file_path = path_template.format(exp)

        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check if the required columns exist
        joint_columns = ["Joint 1", "Joint 2", "Joint 3"]
        if not all(col in df.columns for col in joint_columns):
            print(f"Required joint columns not found in {file_path}")
            continue

        # Calculate distances between consecutive configurations for each joint
        for joint_idx, (joint_col, joint_color) in enumerate(zip(joint_columns, color_set)):
            joint_angles = df[joint_col].to_numpy()
            joint_distances = np.diff(joint_angles)  # Compute differences

            # Plot joint distances
            plt.plot(joint_distances, label=f'Joint {joint_idx + 1}', color=joint_color, linestyle='-', marker='o', linewidth=6, markersize=20)

    # Add legend and show plot
    plt.xticks(fontsize=24, fontweight='bold')
    plt.yticks(fontsize=24, fontweight='bold')

    plt.legend(loc='best', frameon=True, prop={"size":20, "weight": "bold"})
    plt.tight_layout()
    plt.show()








