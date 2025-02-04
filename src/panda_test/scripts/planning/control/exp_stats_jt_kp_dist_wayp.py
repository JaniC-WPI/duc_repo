import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from Robot import RobotTest, PandaReal2D
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os

jt_file_paths_template = [
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/with_obs/nn_25_astar_custom_old/{}/save_distances.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/with_obs/nn_25_astar_custom_old/{}/save_distances.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/with_obs/{}/save_distances.csv'                                                             
]

# Function to read keypoints and joints from a CSV file
def read_keypoints_and_joints(file_path):
    """Reads the keypoints and joints from a CSV file."""
    df = pd.read_csv(file_path)

    # Extract keypoints
    x_keypoints = df.iloc[:, 1:18:2].to_numpy()  # Odd columns
    y_keypoints = df.iloc[:, 2:19:2].to_numpy()  # Even columns

    # print("X kp", x_keypoints)
    # print("Y kp", y_keypoints)

    x_kp_indexed = x_keypoints[:, [1, 3, 4, 6, 7, 8]]
    y_kp_indexed = y_keypoints[:, [1, 3, 4, 6, 7, 8]]

    # print("X kp indexed", x_kp_indexed)
    # print("Y kp indexed", y_kp_indexed)

    # Select specific keypoints (new indices: 0, 3, 4, 5, 6, 7)
    selected_keypoints = np.dstack((x_kp_indexed, y_kp_indexed)).reshape(-1,6,2)

    # print(selected_keypoints.shape)

    # Extract joint angles
    joints = df[["Joint 1", "Joint 2", "Joint 3"]].to_numpy()

    return selected_keypoints, joints

def compute_distance(config1, config2):
    """Compute the Euclidean distance between two configurations."""
    return sum(np.linalg.norm(np.array(kp1) - np.array(kp2)) for kp1, kp2 in zip(config1, config2))

def calculate_experiment_stats(file_path):
    """Calculate total distances and the number of goal configurations from the experiment data."""
    # Read keypoints and joints using the provided function
    keypoints, joints = read_keypoints_and_joints(file_path)
    
    total_keypoint_distance = 0
    total_joint_distance = 0

    # Iterate over all configurations to calculate distances
    for i in range(len(keypoints) - 1):
        # Compute keypoint distance between consecutive configurations
        keypoint_distance = compute_distance(keypoints[i], keypoints[i + 1])
        total_keypoint_distance += keypoint_distance

        # Compute joint distance between consecutive configurations
        joint_distance = np.linalg.norm(joints[i] - joints[i + 1])
        total_joint_distance += joint_distance

    # Count goal configurations (exclude Config_0)
    num_goal_configurations = len(keypoints) - 1  # Exclude Config_0

    return total_keypoint_distance, total_joint_distance, num_goal_configurations
    
labels = ["Ground Truth", "Learned", "Image Space"]

    
# exp_no = [1,2,3,4,6,8,9,10,13,14,15,16,17,18,19,20] # exp with no obs
exp_no = [1, 2, 8, 9, 10, 14, 17, 18, 19, 20] # exps for with obs
# exp_no = [8]

# Initialize storage for experiment statistics
experiment_stats = []

# Iterate over each roadmap type and experiment
for path_template, label in zip(jt_file_paths_template, labels):
    for exp in exp_no:
        file_path = path_template.format(exp)
        if not os.path.exists(file_path):
            continue

        # Calculate statistics for the experiment
        total_kp_dist, total_joint_dist, num_goals = calculate_experiment_stats(file_path)

        # Store the results
        experiment_stats.append({
            "Experiment": exp,
            "Roadmap": label,
            "Total Keypoint Distance": total_kp_dist,
            "Total Joint Distance": total_joint_dist,
            "Number of Goal Configurations": num_goals
        })

# Convert the results to a DataFrame and save as CSV
experiment_stats_df = pd.DataFrame(experiment_stats)
print(experiment_stats_df)

# Save the results to a CSV file
experiment_stats_df.to_csv("/media/jc-merlab/Crucial X9/paper_data/experiment_stats_computed_with_obs.csv", index=False)
print("Experiment statistics saved to 'experiment_stats_computed.csv'")