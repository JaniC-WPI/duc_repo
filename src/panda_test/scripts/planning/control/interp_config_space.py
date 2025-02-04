import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# Function to read keypoints and joints from a CSV file
def read_keypoints_and_joints(file_path):
    """Reads the keypoints and joints from a CSV file."""
    df = pd.read_csv(file_path)

    # Extract keypoints
    x_keypoints = df.iloc[:, 1:18:2].to_numpy()  # Odd columns
    y_keypoints = df.iloc[:, 2:19:2].to_numpy()  # Even columns

    print("X kp", x_keypoints)
    print("Y kp", y_keypoints)

    x_kp_indexed = x_keypoints[:, [1, 3, 4, 6, 7, 8]]
    y_kp_indexed = y_keypoints[:, [1, 3, 4, 6, 7, 8]]

    print("X kp indexed", x_kp_indexed)
    print("Y kp indexed", y_kp_indexed)

    # Select specific keypoints (new indices: 0, 3, 4, 5, 6, 7)
    selected_keypoints = np.dstack((x_kp_indexed, y_kp_indexed)).reshape(-1,6,2)

    print(selected_keypoints.shape)

    # Extract joint angles
    joints = df[["Joint 1", "Joint 2", "Joint 3"]].to_numpy()

    return selected_keypoints, joints

# Function to interpolate joints between two configurations
def interpolate_joints(joint_start, joint_end, num_interpolations=2):
    """Interpolates joint angles between two configurations."""
    return np.linspace(joint_start, joint_end, num_interpolations)

# Function to compute forward kinematics (FK) based on the robot definition
def forward_kinematics(theta, link_lengths):
    """Computes the keypoints in Cartesian space using forward kinematics."""
    # x, y = 0, 0
    # keypoints = []

    x1, y1 = link_lengths[0] * np.cos(theta[0]), link_lengths[0] * np.sin(theta[0])
    x2, y2 = x1 + link_lengths[1] * np.cos(theta[0]+np.pi/2), y1 + link_lengths[1] * np.sin(theta[0]+np.pi/2)
    x3, y3 = x2 + link_lengths[2] * np.cos(theta[0]+np.pi/2+theta[1]), y2 + link_lengths[2] * np.sin(theta[0]+np.pi/2+theta[1])
    x4, y4 = x3 + link_lengths[3] * np.cos(theta[0]+np.pi/2+theta[1]+theta[2]), y3 + link_lengths[3] * np.sin(theta[0]+np.pi/2+theta[1]+theta[2])
    x5, y5 = x4 + link_lengths[4] * np.cos(theta[0]+np.pi/2+theta[1]+theta[2]+np.pi/2), y4 + link_lengths[4] * np.sin(theta[0]+np.pi/2+theta[1]+theta[2]+np.pi/2)
    
    keypoints = [[252, -311], [x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]

    return np.array(keypoints)

def visualize_interpolated_keypoints(original_keypoints, interpolated_keypoints):
    """Visualizes the original and interpolated keypoints."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    for i, og_config in enumerate(original_keypoints):
        # Plot original keypoints
        ax.scatter(og_config[:, 0], -og_config[:, 1], color='red', label='Original Keypoints', s=70, edgecolor='black')
        # Draw lines between original keypoints
        ax.plot(og_config[:, 0], -og_config[:, 1], color='red', linestyle='-', linewidth=1, label='Original Keypoint Chain')

    # Plot interpolated keypoints
    for i, ip_config in enumerate(interpolated_keypoints):
        ax.scatter(ip_config[:, 0], ip_config[:, 1], color='blue', label='Interpolated Keypoints', s=70, edgecolor='black')
        # Label the keypoints in order
        for idx, (x, y) in enumerate(ip_config):
            ax.text(
                x, y, f'{idx}', color='black', fontsize=8, ha='center', va='center'
            )
        
        # Draw lines between interpolated keypoints
        ax.plot(ip_config[:, 0], ip_config[:, 1], color='blue', linestyle='--', linewidth=1, label='Interpolated Keypoint Chain')
        
    for i in range(0, len(interpolated_keypoints), 8):  # Step size of 2
        ip_config = interpolated_keypoints[i]
        print(ip_config)
        ax.scatter(
            ip_config[:, 0],
            ip_config[:, 1],
            color='blue',
            label='Interpolated Keypoints' if i == 0 else "",
            s=70,
            edgecolor='black'
        )
        # Label the keypoints in order
        for idx, (x, y) in enumerate(ip_config):
            ax.text(
                x, y, f'{idx}', color='black', fontsize=8, ha='center', va='center'
            )
        # Draw lines between interpolated keypoints
        if i + 1 < len(interpolated_keypoints):  # Ensure there is a next point to draw the line
            next_ip_config = interpolated_keypoints[i + 1]
            ax.plot(
                [ip_config[:, 0], next_ip_config[:, 0]],
                [ip_config[:, 1], next_ip_config[:, 1]],
                color='blue',
                linestyle='-',
                linewidth=1,
                label='Interpolated Keypoint Chain' if i == 0 else ""
            )

    # Labels and legend
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('Visualization of Interpolated Keypoints', fontsize=14)
    # ax.legend()
    ax.set_xlim(0, 640)
    ax.set_ylim(-480, 250)
    plt.show()

# Function to visualize the interpolated joints
def visualize_interpolation(joints, interpolated_joints):
    """Visualizes the joint configurations in 3D space."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='red', s=100, label='Original Joints')

    # Plot interpolated joints
    for segment in interpolated_joints:
        ax.scatter(segment[:, 0], segment[:, 1], segment[:, 2], color='blue', alpha=0.6, s=100)

    # Labels and legend
    ax.set_xlabel('Joint 1', fontsize=12)
    ax.set_ylabel('Joint 2', fontsize=12)
    ax.set_zlabel('Joint 3', fontsize=12)
    ax.set_title('3D Visualization of Joint Interpolation', fontsize=14)
    ax.legend()
    plt.show()

def calculate_link_lengths(keypoints):
    """Calculates link lengths given a set of keypoints."""
    keypoints = np.array(keypoints)
    return np.linalg.norm(np.diff(keypoints, axis=0), axis=1)

# Base file paths (use `{}` as a placeholder for the experiment number)
jt_file_paths_template = [
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/no_obs/nn_25_astar_custom_old/{}/save_distances.csv'
    # '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/no_obs/nn_25_astar_custom_old/{}/save_distances.csv',
    # '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/no_obs/{}/save_distances.csv'                                                             
]

actual_keypoints = np.array([
    [252, 311],
    [178, 221],
    [201, 201],
    [278, 76],
    [278, 44],
    [318, 45]
])

actual_link_lengths = calculate_link_lengths(actual_keypoints)

print(actual_link_lengths)

link_lengths = [18.35863442, 30.81953253, 147.86035955, 33.23727959, 41.54659949]
# joints = [-0.705008516644936, -1.46633148726214, 2.29688506111637]
joints = [-0.0028, 0, 0]

computed_keypoints = forward_kinematics(joints, link_lengths)

print(computed_keypoints)

comp_link_lengths = calculate_link_lengths(computed_keypoints)

print(comp_link_lengths)

# Visualization
plt.figure(figsize=(8, 6))
# Plot actual keypoints
plt.plot(actual_keypoints[:, 0], -actual_keypoints[:, 1], 'ro-', label='Actual Keypoints', markersize=8)
# Plot computed keypoints
plt.plot(computed_keypoints[:, 0], computed_keypoints[:, 1], 'bo--', label='Computed Keypoints (DH)', markersize=8)

# Annotate the keypoints
for i, (x, y) in enumerate(actual_keypoints):
    plt.text(x, -y, f'P{i+1}', fontsize=9, color='red', ha='right')

for i, (x, y) in enumerate(computed_keypoints):
    plt.text(x, y, f'C{i+1}', fontsize=9, color='blue', ha='left')

# Add labels and legend
plt.title('Comparison of Actual and Computed Keypoints (Corrected DH)', fontsize=14)
plt.xlabel('X (pixels)', fontsize=12)
plt.ylabel('Y (pixels)', fontsize=12)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True)
plt.legend()
plt.show()

exp_no = [1]

# Link lengths (fixed)
link_lengths = [118.35863442, 30.81953253, 147.86035955, 33.23727959, 41.54659949]

# Process each experiment
for path_template in jt_file_paths_template:
    for exp in exp_no:
        file_path = path_template.format(exp)
        if not os.path.exists(file_path):
            continue

        # Read keypoints and joints
        keypoints, joints = read_keypoints_and_joints(file_path)

        # Interpolate joints between configurations
        interpolated_joints = []
        for i in range(len(joints) - 1):
            segment = interpolate_joints(joints[i], joints[i + 1], num_interpolations=1)
            interpolated_joints.append(segment)

        # Visualize the interpolation
        visualize_interpolation(joints, interpolated_joints)

        # # Interpolate joints between configurations
        interpolated_keypoints = []
        for i in range(len(joints) - 1):
            segment = interpolate_joints(joints[i], joints[i + 1], num_interpolations=1)
            for interp_joint in segment:
                interpolated_keypoints.append(forward_kinematics(interp_joint, link_lengths))
        
        # interpolated_keypoints = np.array(interpolated_keypoints)
        
        # # print("Original Keypoints", keypoints)
        # print("Interpolated Keypoints", interpolated_keypoints)

        # Visualize interpolated keypoints
        visualize_interpolated_keypoints(keypoints, interpolated_keypoints)











