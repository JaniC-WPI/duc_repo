import os
import json
import numpy as np
from math import acos, degrees

def read_keypoints_from_json(json_path, indices_to_read):
    """Reads keypoints from the JSON file at the specified indices."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    keypoints = data.get("keypoints", [])
    return [keypoints[i][0][:2] for i in indices_to_read]

def calculate_link_lengths(keypoints):
    """Calculates link lengths given a set of keypoints."""
    keypoints = np.array(keypoints)
    return np.linalg.norm(np.diff(keypoints, axis=0), axis=1)

def calculate_angle(v1, v2):
    """Calculates the angle between two vectors in degrees."""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Handle numerical inaccuracies
    return degrees(acos(cos_theta))

def calculate_angles_at_points(keypoints):
    """Calculates angles at the specified points."""
    keypoints = np.array(keypoints)
    angles = []
    for i, index in enumerate([1, 4]):  # Indices 1 and 4 correspond to 3 and 7 in original
        vector1 = keypoints[index] - keypoints[index - 1]
        vector2 = keypoints[index + 1] - keypoints[index]
        angles.append(calculate_angle(vector1, vector2))
    return angles

def process_folder(folder_path):
    """Processes all JSON files in a folder."""
    indices_to_read = [1, 3, 4, 6, 7, 8]  # Indices to read from keypoints
    all_link_lengths = []
    all_angles_at_points = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            json_path = os.path.join(folder_path, file_name)
            keypoints = read_keypoints_from_json(json_path, indices_to_read)

            # Calculate link lengths
            link_lengths = calculate_link_lengths(keypoints)
            all_link_lengths.append(link_lengths)

            # Calculate angles at points
            angles = calculate_angles_at_points(keypoints)
            all_angles_at_points.append(angles)

    # Convert to numpy arrays for easier averaging
    all_link_lengths = np.array(all_link_lengths)
    all_angles_at_points = np.array(all_angles_at_points)

    # Calculate averages
    average_link_lengths = np.mean(all_link_lengths, axis=0)
    average_angles_at_points = np.mean(all_angles_at_points, axis=0)

    return all_link_lengths, average_link_lengths, all_angles_at_points, average_angles_at_points

# Example usage
folder_path = "/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/planning_kprcnn_rearranged/"
all_link_lengths, average_link_lengths, all_angles_at_points, average_angles_at_points = process_folder(folder_path)

# print("Link Lengths for Each File:")
# print(all_link_lengths)
print("\nAverage Link Lengths:")
print(average_link_lengths)
# print("\nAngles at Points for Each File:")
# print(all_angles_at_points)
print("\nAverage Angles at Points:")
print(average_angles_at_points)

import numpy as np
import matplotlib.pyplot as plt

def interpolate_joint_configurations(configurations, num_interpolations=100):
    """
    Interpolates between successive joint configurations.

    Parameters:
        configurations (list of list of float): List of joint configurations.
        num_interpolations (int): Number of intermediate points between each pair of configurations.

    Returns:
        np.ndarray: Array of interpolated configurations.
    """
    all_interpolated = []

    for i in range(len(configurations) - 1):
        start_config = np.array(configurations[i])
        end_config = np.array(configurations[i + 1])

        # Interpolate linearly between start_config and end_config
        interpolated = np.linspace(start_config, end_config, num_interpolations + 2)
        all_interpolated.append(interpolated[:-1])  # Exclude the last point to avoid duplication

    # Add the final configuration
    all_interpolated.append([configurations[-1]])

    # Combine all interpolated points
    return np.vstack(all_interpolated)

def visualize_joint_space(configurations, interpolated_configurations):
    """
    Visualizes the joint configurations in 3D space.

    Parameters:
        configurations (list of list of float): Original joint configurations.
        interpolated_configurations (np.ndarray): Interpolated joint configurations.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert configurations to NumPy array for easier manipulation
    configurations = np.array(configurations)

    # Plot original configurations
    ax.scatter(configurations[:, 0], configurations[:, 1], configurations[:, 2],
               color='red', s=100, label='Original Configurations')

    # Plot interpolated configurations
    ax.scatter(interpolated_configurations[:, 0], interpolated_configurations[:, 1], interpolated_configurations[:, 2],
               color='blue', s=20, label='Interpolated Configurations')

    # Highlight original configurations with larger markers
    ax.scatter(configurations[:, 0], configurations[:, 1], configurations[:, 2],
               color='red', s=200, edgecolor='black')

    # Labels and legend
    ax.set_xlabel('Joint 1', fontsize=12)
    ax.set_ylabel('Joint 2', fontsize=12)
    ax.set_zlabel('Joint 3', fontsize=12)
    ax.set_title('3D Visualization of Joint Configurations', fontsize=14)
    ax.legend()

    plt.show()

# Input joint configurations
joint_configurations = [
    [-0.870795, -1.44845, 2.20395],
    [-0.705173152786279, -1.62948008523155, 2.36142950032406],
    [-0.388997423598641, -1.98781081152743, 2.36187350707586],
    [-0.072936911293516, -1.62901536948783, 2.31292020576528],
    [0.243332407380626, -1.37700142454318, 2.36530701472566],
    [0.243308746910932, -1.36211298632417, 2.36530812062158],
    [0.267307, -1.38323, 2.58668]
]

# Interpolate between configurations
num_interpolations = 10
interpolated_configurations = interpolate_joint_configurations(joint_configurations, num_interpolations)

# Visualize the results
visualize_joint_space(joint_configurations, interpolated_configurations)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, BSpline, make_interp_spline

def nonlinear_interpolate_joint_configurations(configurations, num_interpolations=100):
    """
    Non-linear interpolation between successive joint configurations using cubic splines.

    Parameters:
        configurations (list of list of float): List of joint configurations.
        num_interpolations (int): Number of intermediate points per segment.

    Returns:
        np.ndarray: Array of interpolated configurations.
    """
    configurations = np.array(configurations)
    num_joints = configurations.shape[1]
    interpolated_configurations = []

    for joint_index in range(num_joints):
        # Extract joint values for the current joint
        joint_values = configurations[:, joint_index]

        # Generate cubic spline for the joint values
        spline = make_interp_spline(range(len(joint_values)), joint_values)

        # Evaluate the spline to generate intermediate points
        interpolated_joint_values = spline(np.linspace(0, len(joint_values) - 1, (len(joint_values) - 1) * num_interpolations + 1))
        interpolated_configurations.append(interpolated_joint_values)

    # Combine all interpolated joint values
    return np.array(interpolated_configurations).T

def visualize_3d_joint_space(original_configurations, interpolated_configurations):
    """
    Visualizes the joint configurations in 3D space.

    Parameters:
        original_configurations (list of list of float): Original joint configurations.
        interpolated_configurations (np.ndarray): Interpolated joint configurations.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert configurations to NumPy array for easier manipulation
    original_configurations = np.array(original_configurations)

    # Plot original configurations
    ax.scatter(original_configurations[:, 0], original_configurations[:, 1], original_configurations[:, 2],
               color='red', s=100, label='Original Configurations')

    # Plot interpolated configurations
    ax.scatter(interpolated_configurations[:, 0], interpolated_configurations[:, 1], interpolated_configurations[:, 2],
               color='blue', s=20, label='Interpolated Configurations')

    # Highlight original configurations with larger markers
    ax.scatter(original_configurations[:, 0], original_configurations[:, 1], original_configurations[:, 2],
               color='red', s=200, edgecolor='black')

    # Labels and legend
    ax.set_xlabel('Joint 1', fontsize=12)
    ax.set_ylabel('Joint 2', fontsize=12)
    ax.set_zlabel('Joint 3', fontsize=12)
    ax.set_title('3D Visualization of Non-Linear Interpolation in Joint Space', fontsize=14)
    ax.legend()

    plt.show()

# Input joint configurations
joint_configurations = [
    [-0.870795, -1.44845, 2.20395],
    [-0.705173152786279, -1.62948008523155, 2.36142950032406],
    [-0.388997423598641, -1.98781081152743, 2.36187350707586],
    [-0.072936911293516, -1.62901536948783, 2.31292020576528],
    [0.243332407380626, -1.37700142454318, 2.36530701472566],
    [0.243308746910932, -1.36211298632417, 2.36530812062158],
    [0.267307, -1.38323, 2.58668]
]

# Perform non-linear interpolation
num_interpolations = 10
interpolated_configurations = nonlinear_interpolate_joint_configurations(joint_configurations, num_interpolations)

# Visualize the results
visualize_3d_joint_space(joint_configurations, interpolated_configurations)

def nonlinear_interpolate_joint_configurations(configurations, num_interpolations=100):
    """
    Non-linear interpolation between successive joint configurations using B-splines.

    Parameters:
        configurations (list of list of float): List of joint configurations.
        num_interpolations (int): Number of intermediate points per segment.

    Returns:
        np.ndarray: Array of interpolated configurations.
    """
    configurations = np.array(configurations)
    num_joints = configurations.shape[1]
    interpolated_configurations = []

    for joint_index in range(num_joints):
        # Extract joint values for the current joint
        joint_values = configurations[:, joint_index]

        # Generate uniform knots for the B-spline
        num_points = len(joint_values)
        degree = 3  # Degree of the B-spline
        knots = np.concatenate(([0] * degree, np.arange(num_points - degree + 1), [num_points - degree] * degree))

        # Create the B-spline
        spline = BSpline(knots, joint_values, degree)

        # Evaluate the spline to generate intermediate points
        interpolated_joint_values = spline(np.linspace(0, num_points - degree, (num_points - 1) * num_interpolations + 1))
        interpolated_configurations.append(interpolated_joint_values)

    # Combine all interpolated joint values
    return np.array(interpolated_configurations).T

def visualize_3d_joint_space(original_configurations, interpolated_configurations):
    """
    Visualizes the joint configurations in 3D space.

    Parameters:
        original_configurations (list of list of float): Original joint configurations.
        interpolated_configurations (np.ndarray): Interpolated joint configurations.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert configurations to NumPy array for easier manipulation
    original_configurations = np.array(original_configurations)

    # Plot original configurations
    ax.scatter(original_configurations[:, 0], original_configurations[:, 1], original_configurations[:, 2],
               color='red', s=100, label='Original Configurations')

    # Plot interpolated configurations
    ax.scatter(interpolated_configurations[:, 0], interpolated_configurations[:, 1], interpolated_configurations[:, 2],
               color='blue', s=20, label='Interpolated Configurations')

    # Highlight original configurations with larger markers
    ax.scatter(original_configurations[:, 0], original_configurations[:, 1], original_configurations[:, 2],
               color='red', s=200, edgecolor='black')

    # Labels and legend
    ax.set_xlabel('Joint 1', fontsize=12)
    ax.set_ylabel('Joint 2', fontsize=12)
    ax.set_zlabel('Joint 3', fontsize=12)
    ax.set_title('3D Visualization of Non-Linear Interpolation in Joint Space (B-Spline)', fontsize=14)
    ax.legend()

    plt.show()

# Input joint configurations
joint_configurations = [
    [-0.870795, -1.44845, 2.20395],
    [-0.705173152786279, -1.62948008523155, 2.36142950032406],
    [-0.388997423598641, -1.98781081152743, 2.36187350707586],
    [-0.072936911293516, -1.62901536948783, 2.31292020576528],
    [0.243332407380626, -1.37700142454318, 2.36530701472566],
    [0.243308746910932, -1.36211298632417, 2.36530812062158],
    [0.267307, -1.38323, 2.58668]
]

# Perform non-linear interpolation
num_interpolations = 100
interpolated_configurations = nonlinear_interpolate_joint_configurations(joint_configurations, num_interpolations)

# Visualize the results
visualize_3d_joint_space(joint_configurations, interpolated_configurations)


