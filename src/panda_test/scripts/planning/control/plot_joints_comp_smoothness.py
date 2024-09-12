import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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
    - avg_distance: The average Euclidean distance between consecutive joint configurations.
    """
    distances = []
    for i in range(len(joint1) - 1):
        dist = np.linalg.norm([joint1.iloc[i+1] - joint1.iloc[i],
                               joint2.iloc[i+1] - joint2.iloc[i],
                               joint3.iloc[i+1] - joint3.iloc[i]])
        distances.append(dist)
    
    avg_distance = np.mean(distances)
    return avg_distance

def plot_joint_configurations(ax, file_path, label, color, start_marker='o', goal_marker='X'):
    """
    Reads a CSV file containing joint configurations and plots them in 3D, while calculating smoothness.

    Args:
    - ax (Axes3D): The 3D axes object to plot on.
    - file_path (str): Path to the CSV file.
    - label (str): Label for the plot legend.
    - color (str): Color of the plot line.
    - start_marker (str): Marker style for the start joint.
    - goal_marker (str): Marker style for the goal joint.

    Returns:
    - smoothness (float): A measure of the trajectory's smoothness.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Extract joint angles
    joint1 = df['Joint 1']
    joint2 = df['Joint 2']
    joint3 = df['Joint 3']

    # Plot joint configurations in 3D
    ax.plot(joint1, joint2, joint3, label=label, color=color)
    ax.scatter(joint1, joint2, joint3, color=color)

    # Mark the start and goal joint angles
    if label == 'Ground Truth':
        ax.scatter(joint1.iloc[0], joint2.iloc[0], joint3.iloc[0], color='green', marker=start_marker, s=100, label='Start')
        ax.scatter(joint1.iloc[-1], joint2.iloc[-1], joint3.iloc[-1], color='purple', marker=goal_marker, s=100, label='Goal')

    # Calculate total variation for each joint
    variation_joint1 = calculate_total_variation(joint1)
    variation_joint2 = calculate_total_variation(joint2)
    variation_joint3 = calculate_total_variation(joint3)
    avg_variation = (variation_joint1 + variation_joint2 + variation_joint3) / 3

    # Calculate average distance between consecutive joint configurations
    avg_distance = calculate_average_distance(joint1, joint2, joint3)

    print(f"{label} Trajectory Smoothness (Total Variation): {avg_variation:.4f}")
    print(f"{label} Trajectory Smoothness (Average Distance): {avg_distance:.4f}")

    return avg_variation, avg_distance

def main():
    # Paths to the CSV files
    file_path1 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/with_obs/302/save_distances.csv'
    file_path2 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/with_obs/302/save_distances.csv'
    file_path3 =  '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/with_obs/302/save_distances.csv'

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot joint configurations and calculate smoothness for each trajectory
    smoothness_custom = plot_joint_configurations(ax, file_path1, label='Custom', color='blue')
    smoothness_euclidean = plot_joint_configurations(ax, file_path2, label='Euclidean', color='red')
    smoothness_ground_truth = plot_joint_configurations(ax, file_path3, label='Ground Truth', color='green')

    # Set labels
    ax.set_xlabel('Joint 1')
    ax.set_ylabel('Joint 2')
    ax.set_zlabel('Joint 3')
    ax.set_title('3D Joint Configurations Comparison')
    ax.legend(loc='upper right')

    # Save the plot
    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_plots/with_obs/jt_dist_3.png', dpi=300)

    # Show the plot
    plt.show()

    # Print overall smoothness for each trajectory
    print(f"Custom Trajectory Smoothness (Total Variation, Avg Distance): {smoothness_custom}")
    print(f"Euclidean Trajectory Smoothness (Total Variation, Avg Distance): {smoothness_euclidean}")
    print(f"Ground Truth Trajectory Smoothness (Total Variation, Avg Distance): {smoothness_ground_truth}")

# Run the main function
if __name__ == "__main__":
    main()
