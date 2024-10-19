import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.lines import Line2D
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def calculate_jerk(joint_positions):
    """
    Calculates the jerk (rate of change of acceleration) for a given sequence of joint positions.
    
    Args:
    - joint_positions (pd.Series): A pandas series of joint positions (e.g., Joint 1, Joint 2, Joint 3).
    
    Returns:
    - jerk: A measure of the smoothness of the trajectory.
    """
    # Calculate first and second derivatives (acceleration and jerk)
    velocity = np.gradient(joint_positions)  # First derivative
    acceleration = np.gradient(velocity)     # Second derivative (acceleration)
    jerk = np.gradient(acceleration)         # Third derivative (jerk)
    
    # Return the norm of the jerk as a measure of smoothness
    return np.linalg.norm(jerk)

def calculate_total_variation(joint_positions):
    """
    Calculates the total variation for a given sequence of joint positions.
    The smaller the variation, the smoother the trajectory.

    Args:
    - joint_positions (pd.Series): A pandas series of joint positions (e.g., Joint 1, Joint 2, Joint 3).
    
    Returns:
    - total_variation: The sum of absolute differences between consecutive joint positions.
    """
    variation = np.abs(np.diff(joint_positions))
    total_variation = np.sum(variation)
    return total_variation

def calculate_average_distance(joint1, joint2, joint3):
    """
    Calculates the average Euclidean distance between consecutive joint configurations.
    
    Args:
    - joint1, joint2, joint3 (pd.Series): Joint angle series for each joint.
    
    Returns:
    - total_distances: The total Euclidean distance between consecutive joint configurations.
    """
    distances = []
    for i in range(len(joint1) - 1):
        dist = np.linalg.norm([joint1.iloc[i+1] - joint1.iloc[i],
                               joint2.iloc[i+1] - joint2.iloc[i],
                               joint3.iloc[i+1] - joint3.iloc[i]])
        distances.append(dist)
    
    total_distances = np.sum(distances)
    return total_distances

def plot_joint_configurations(ax, file_path, label, color, start_marker='o', goal_marker='X'):
    df = pd.read_csv(file_path)
    joint1 = df['Joint 1']
    joint2 = df['Joint 2']
    joint3 = df['Joint 3']

    # Plot the 3D trajectory
    line_handle, = ax.plot(joint1, joint2, joint3, label=label, color=color)  # Store line handle
    ax.scatter(joint1, joint2, joint3, color=color)

    # Add start and goal markers
    start_marker_handle = ax.scatter(joint1.iloc[0], joint2.iloc[0], joint3.iloc[0], color='green', marker=start_marker, s=100)
    goal_marker_handle = ax.scatter(joint1.iloc[-1], joint2.iloc[-1], joint3.iloc[-1], color='purple', marker=goal_marker, s=100)
    
    # Calculate variations
    variation_joint1 = calculate_total_variation(joint1)
    variation_joint2 = calculate_total_variation(joint2)
    variation_joint3 = calculate_total_variation(joint3)
    avg_variation = (variation_joint1 + variation_joint2 + variation_joint3) / 3

    # Calculate total distances
    total_distances = calculate_average_distance(joint1, joint2, joint3)

    # Calculate jerk
    jerk_joint1 = calculate_jerk(joint1)
    jerk_joint2 = calculate_jerk(joint2)
    jerk_joint3 = calculate_jerk(joint3)
    avg_jerk = (jerk_joint1 + jerk_joint2 + jerk_joint3) / 3

    # Print variation and jerk information
    print(f"{label} Trajectory Smoothness (Variation First Joint): {variation_joint1:.4f}")
    print(f"{label} Trajectory Smoothness (Variation Second Joint): {variation_joint2:.4f}")
    print(f"{label} Trajectory Smoothness (Variation Third Joint): {variation_joint3:.4f}")
    print(f"{label} Trajectory Smoothness (Jerk First Joint): {jerk_joint1:.4f}")
    print(f"{label} Trajectory Smoothness (Jerk Second Joint): {jerk_joint2:.4f}")
    print(f"{label} Trajectory Smoothness (Jerk Third Joint): {jerk_joint3:.4f}")

    return variation_joint3, total_distances, jerk_joint3, start_marker_handle, goal_marker_handle, line_handle

def calculate_dtw_distance(joint1_1, joint2_1, joint3_1, joint1_2, joint2_2, joint3_2):
    """
    Calculates the Dynamic Time Warping (DTW) distance between two joint configurations.
    
    Args:
    - joint1_1, joint2_1, joint3_1 (pd.Series): First set of joint angle series.
    - joint1_2, joint2_2, joint3_2 (pd.Series): Second set of joint angle series.
    
    Returns:
    - dtw_distance: The DTW distance between the two trajectories.
    - path: The optimal alignment path between the two sequences.
    """
    # Combine joint positions into a 2D array for each trajectory
    series1 = np.array([joint1_1, joint2_1, joint3_1]).T
    series2 = np.array([joint1_2, joint2_2, joint3_2]).T

    # Compute the DTW distance and the optimal alignment path
    dtw_distance, path = fastdtw(series1, series2, dist=euclidean)
    return dtw_distance, path

def plot_dtw_alignment(joint1_1, joint2_1, joint3_1, joint1_2, joint2_2, joint3_2, path, label1, label2):
    """
    Plots the DTW alignment between two sets of joint configurations for all three joints.
    
    Args:
    - joint1_1, joint2_1, joint3_1 (pd.Series): First set of joint angle series.
    - joint1_2, joint2_2, joint3_2 (pd.Series): Second set of joint angle series.
    - path: The alignment path calculated by DTW.
    - label1, label2 (str): Labels for the two series in the plot.
    """
    # Combine joint configurations for plotting
    series1 = np.array([joint1_1, joint2_1, joint3_1]).T
    series2 = np.array([joint1_2, joint2_2, joint3_2]).T
    
    # Create a plot to visualize the alignment for all three joints
    plt.figure(figsize=(12, 8))
    
    # Plot Joint 1
    plt.plot(series1[:, 0], label=f"{label1} Joint 1", color='blue')
    plt.plot(series2[:, 0], label=f"{label2} Joint 1", color='red')
    
    # Plot Joint 2
    plt.plot(series1[:, 1], label=f"{label1} Joint 2", color='green')
    plt.plot(series2[:, 1], label=f"{label2} Joint 2", color='orange')
    
    # Plot Joint 3
    plt.plot(series1[:, 2], label=f"{label1} Joint 3", color='purple')
    plt.plot(series2[:, 2], label=f"{label2} Joint 3", color='brown')

    # Plot DTW alignment path for all joints
    for (i, j) in path:
        plt.plot([i, j], [series1[i, 0], series2[j, 0]], color='black', linestyle='--', linewidth=0.8)  # Joint 1
        plt.plot([i, j], [series1[i, 1], series2[j, 1]], color='black', linestyle='--', linewidth=0.8)  # Joint 2
        plt.plot([i, j], [series1[i, 2], series2[j, 2]], color='black', linestyle='--', linewidth=0.8)  # Joint 3

    # Set the plot labels and legend
    plt.title(f"DTW Alignment between {label1} and {label2}")
    plt.xlabel('Time Step')
    plt.ylabel('Joint Values')
    plt.legend()
    plt.show()

def load_and_calculate_dtw(file_path1, file_path2, file_path3):
    """
    Load joint data from CSV files and calculate DTW distances between different roadmaps.
    """
    # Load the joint data for DTW calculations
    df_custom = pd.read_csv(file_path1)
    df_euclidean = pd.read_csv(file_path2)
    df_ground_truth = pd.read_csv(file_path3)

    # Extract joint positions for each roadmap
    joint1_custom, joint2_custom, joint3_custom = df_custom['Joint 1'], df_custom['Joint 2'], df_custom['Joint 3']
    joint1_euclidean, joint2_euclidean, joint3_euclidean = df_euclidean['Joint 1'], df_euclidean['Joint 2'], df_euclidean['Joint 3']
    joint1_ground_truth, joint2_ground_truth, joint3_ground_truth = df_ground_truth['Joint 1'], df_ground_truth['Joint 2'], df_ground_truth['Joint 3']

    # Calculate DTW distances and paths between the roadmaps
    dtw_custom_euclidean, path_custom_euclidean = calculate_dtw_distance(joint1_custom, joint2_custom, joint3_custom,
                                                                         joint1_euclidean, joint2_euclidean, joint3_euclidean)
    dtw_custom_ground_truth, path_custom_ground_truth = calculate_dtw_distance(joint1_custom, joint2_custom, joint3_custom,
                                                                               joint1_ground_truth, joint2_ground_truth, joint3_ground_truth)
    dtw_euclidean_ground_truth, path_euclidean_ground_truth = calculate_dtw_distance(joint1_euclidean, joint2_euclidean, joint3_euclidean,
                                                                                     joint1_ground_truth, joint2_ground_truth, joint3_ground_truth)

    # Print DTW distances
    print(f"DTW Distance (Custom vs Euclidean): {dtw_custom_euclidean:.4f}")
    print(f"DTW Distance (Custom vs Ground Truth): {dtw_custom_ground_truth:.4f}")
    print(f"DTW Distance (Euclidean vs Ground Truth): {dtw_euclidean_ground_truth:.4f}")

    # Plot DTW alignment for Custom vs Ground Truth and Euclidean vs Ground Truth for all joints
    plot_dtw_alignment(joint1_custom, joint2_custom, joint3_custom, joint1_ground_truth, joint2_ground_truth, joint3_ground_truth, path_custom_ground_truth, 'Custom', 'Ground Truth')
    plot_dtw_alignment(joint1_euclidean, joint2_euclidean, joint3_euclidean, joint1_ground_truth, joint2_ground_truth, joint3_ground_truth, path_euclidean_ground_truth, 'Euclidean', 'Ground Truth')

def main():
    # Paths to the CSV files
    file_path1 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/no_obs/nn_25_astar_custom/8/save_distances.csv'
    file_path2 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/no_obs/nn_25_astar_custom/8/save_distances.csv'
    file_path3 =  '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/no_obs/nn_25_astar_custom/8/save_distances.csv'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get variations and distances for each trajectory
    smoothness_custom, custom_distances, _, start_handle_custom, goal_handle_custom, line_handle_custom = plot_joint_configurations(ax, file_path1, label='Custom', color='blue')
    smoothness_euclidean, euclidean_distances, _, start_handle_euclidean, goal_handle_euclidean, line_handle_euclidean = plot_joint_configurations(ax, file_path2, label='Euclidean', color='red')
    smoothness_ground_truth, ground_truth_distances, _, start_handle_ground_truth, goal_handle_ground_truth, line_handle_ground_truth = plot_joint_configurations(ax, file_path3, label='Ground Truth', color='green')

    # Set labels
    ax.set_xlabel('Joint 1')
    ax.set_ylabel('Joint 2')
    ax.set_zlabel('Joint 3')
    ax.set_title('3D Joint Configurations Comparison')

    # Modify the legend to include smoothness data
    custom_label = f"Custom (Dist: {custom_distances:.2f})"
    euclidean_label = f"Euclidean (Dist: {euclidean_distances:.2f})"
    ground_truth_label = f"Ground Truth (Dist: {ground_truth_distances:.2f})"

    # Set the legend with the updated labels
    ax.legend([start_handle_custom, goal_handle_custom, 
               Line2D([0], [0], color='green', label=ground_truth_label),
               Line2D([0], [0], color='blue', label=custom_label),
               Line2D([0], [0], color='red', label=euclidean_label)],
              ['Start', 'Goal', ground_truth_label, custom_label, euclidean_label], loc='upper right')

    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_plots/no_obs/jt_dist_2.png', dpi=300)
    plt.show()

    print(f"Custom Trajectory Smoothness (Avg Variation, Total Distance): {smoothness_custom}")
    print(f"Euclidean Trajectory Smoothness (Avg Variation, Total Distance): {smoothness_euclidean}")
    print(f"Ground Truth Trajectory Smoothness (Avg Variation, Total Distance): {smoothness_ground_truth}")

    # # Load the joint data for DTW calculations
    df_custom = pd.read_csv(file_path1)
    df_euclidean = pd.read_csv(file_path2)
    df_ground_truth = pd.read_csv(file_path3)

    # # Extract joint positions for each roadmap
    joint1_custom, joint2_custom, joint3_custom = df_custom['Joint 1'], df_custom['Joint 2'], df_custom['Joint 3']
    joint1_euclidean, joint2_euclidean, joint3_euclidean = df_euclidean['Joint 1'], df_euclidean['Joint 2'], df_euclidean['Joint 3']
    joint1_ground_truth, joint2_ground_truth, joint3_ground_truth = df_ground_truth['Joint 1'], df_ground_truth['Joint 2'], df_ground_truth['Joint 3']

    # # Calculate DTW distances
    # dtw_custom_euclidean = calculate_dtw_distance(joint1_custom, joint2_custom, joint3_custom,
    #                                               joint1_euclidean, joint2_euclidean, joint3_euclidean)
    # dtw_custom_ground_truth = calculate_dtw_distance(joint1_custom, joint2_custom, joint3_custom,
    #                                                  joint1_ground_truth, joint2_ground_truth, joint3_ground_truth)
    # dtw_euclidean_ground_truth = calculate_dtw_distance(joint1_euclidean, joint2_euclidean, joint3_euclidean,
    #                                                     joint1_ground_truth, joint2_ground_truth, joint3_ground_truth)

    # # Print DTW distances
    # print(f"DTW Distance (Custom vs Euclidean): {dtw_custom_euclidean:.2f}")
    # print(f"DTW Distance (Custom vs Ground Truth): {dtw_custom_ground_truth:.2f}")
    # print(f"DTW Distance (Euclidean vs Ground Truth): {dtw_euclidean_ground_truth:.2f}")

    load_and_calculate_dtw(file_path1, file_path2, file_path3)

if __name__ == "__main__":
    main()