import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib
import os
matplotlib.use('TkAgg')

save_directory = "/media/jc-merlab/Crucial X9/paper_data/trajectory_pics/"

# File paths
file_paths = [
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/ground_truth/astar_latest_no_obs/1/joint_angles.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_no_obs/1/joint_angles.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/euclidean/astar_latest_no_obs/1/joint_angles.csv'
]

jt_file_paths = [
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/no_obs/nn_25_astar_custom_old/1/save_distances.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/no_obs/nn_25_astar_custom_old/1/save_distances.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/no_obs/1/save_distances.csv'
]

kp_file_paths = [
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/ground_truth/astar_latest_no_obs/1/cp.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_no_obs/1/cp.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/euclidean/astar_latest_no_obs/1/cp.csv'
]

# Initialize the goal_configurations list
goal_configurations = []

# Loop through each file and process it
for file_path in jt_file_paths:
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Extract the 'Joint 1', 'Joint 2', 'Joint 3' columns
    joints = df[['Joint 1', 'Joint 2', 'Joint 3']].values.tolist()
    
    # Append the joint values as a list to goal_configurations
    goal_configurations.append(joints)

# Function to calculate nrows and read truncated data
# def read_control_data(file_path, skip_header=751, exclude_last=500):
#     """Read CSV starting at skip_header, skip every 3rd row, and exclude the last rows."""
#     # Get the total number of rows in the file
#     total_rows = sum(1 for _ in open(file_path))  # Count total rows
    
#     # Calculate the range of rows to consider (after header and before last rows)
#     start_row = skip_header
#     end_row = total_rows - exclude_last
    
#     # Generate indices of rows to skip: skip_header + every 3rd row until end_row
#     rows_to_skip = set(range(start_row, end_row, 3))  # Skip every 3rd row
#     rows_to_skip.update(range(skip_header))  # Skip header rows
    
#     # Read the CSV excluding the generated rows
#     data = pd.read_csv(file_path, skiprows=lambda x: x in rows_to_skip)
#     return data
    
# def read_control_data(file_path, skip_header=751, exclude_last=100):
#     """Read CSV starting at skip_header and exclude last rows."""
#     # Get total number of rows in the file
#     total_rows = sum(1 for _ in open(file_path))  # Fast row count
#     nrows = total_rows - skip_header - exclude_last  # Rows to read
    
#     # Read the desired range from the file
#     data = pd.read_csv(file_path, skiprows=skip_header, nrows=nrows)
#     return data

# exclude_last_values = [750, 900, 0]

# # Load the control data for the three files
# control_data = [
#     read_control_data(fp, exclude_last=exclude_last)
#     for fp, exclude_last in zip(file_paths, exclude_last_values)
# ]
    
def get_last_goal_row_count(kp_file):
    """
    Function to determine the row count for the last goal in a given kp_file.
    """
    # Read the kp_file CSV
    kp_df = pd.read_csv(kp_file, skiprows=range(1, 52))
    # Extract the 'goal_column' (first column)
    goal_columns = kp_df.iloc[:, 0].to_numpy()
    # Identify the unique goals
    unique_goals = np.unique(goal_columns)
    print("unique goals", unique_goals)
    # Find the last goal
    last_goal = unique_goals[-1]
    print("last goal", last_goal)
    # Count rows corresponding to the last goal
    last_goal_rows = (goal_columns == last_goal).sum()
    print("Number of rows", last_goal_rows)
    return last_goal_rows

def read_control_data_limited(file_path, last_goal_row_count):
    """
    Reads the control_data CSV but limits the rows for the last goal to 3 * last_goal_row_count.
    """
    # Read the entire CSV
    control_df = pd.read_csv(file_path, skiprows=763, header=None)
    # Extract the 'goal_column' (first column)
    goal_columns = control_df.iloc[:, 0].to_numpy()
    # Identify the unique goals
    unique_goals = np.unique(goal_columns)
    # Find the last goal
    last_goal = unique_goals[-1]
    
    # Get indices of rows corresponding to the last goal
    last_goal_indices = np.where(goal_columns == last_goal)[0]
    # Limit the rows for the last goal to 3 * last_goal_row_count
    limit = 3 * last_goal_row_count
    last_goal_indices_limited = last_goal_indices[:limit]
    print(last_goal_indices_limited)
    
    # Combine rows from earlier goals and the limited rows of the last goal
    rows_to_keep = np.concatenate([np.where(goal_columns != last_goal)[0], last_goal_indices_limited])
    control_df_limited = control_df.iloc[rows_to_keep]

    last_row_index = control_df_limited.index[-1]
    print("Index of the last row in control_df_limited:", last_row_index)
    
    print("Number of rows in control_df_limited:", len(control_df_limited))
    
    return control_df_limited

# Step 1: Get row counts for last goals in kp_file_paths
last_goal_row_counts = [get_last_goal_row_count(kp_file) for kp_file in kp_file_paths]

# Step 2: Use these row counts to read the control_data from file_paths
control_data = [
    read_control_data_limited(fp, row_count)
    for fp, row_count in zip(file_paths, last_goal_row_counts)
]

# Load the control data
# control_data = [pd.read_csv(fp, header=758) for fp in file_paths]
# control_data = [read_control_data(fp) for fp in file_paths]

# Helper functions
def calculate_norm(configurations):
    """Calculates the Euclidean norm of the joint angles for each configuration."""
    return np.linalg.norm(configurations, axis=1)

# Determine the maximum number of goal configurations across all cases
global_x_max = max(len(goals) for goals in goal_configurations)

# Pre-compute the maximum norm across all datasets for consistent y-axis limits
global_y_max = max([np.max(calculate_norm(data.iloc[:, 1:4].to_numpy())) for data in control_data])


def interpolate_norms_continuous(start_config, goal_config, num_points):
    """Interpolates the norms of the start and goal configurations."""
    start_norm = np.linalg.norm(start_config)
    goal_norm = np.linalg.norm(goal_config)
    return np.linspace(start_norm, goal_norm, num_points)

# Colors and labels for plotting
colors=['#40B0A6', '#5D3A9B', '#D41159']
labels = ['Ground Truth', 'Learned', 'Image Space']

# Define consistent x-positions for start and goal points
goal_x_positions = [0]  # Start from 0
for goals in goal_configurations:
    goal_x_positions.extend(np.cumsum([len(goals) for _ in range(1, len(goals))]))

# Plot setup
plt.figure(figsize=(20, 15))

# Process each file
for idx, (data, goals, color, method) in enumerate(zip(control_data, goal_configurations, colors, labels)):
    # Extract goal column and joint angles
    goal_column = data.iloc[:, 0].values
    control_joints = data.iloc[:, 1:4].to_numpy()
    
    # Calculate actual norms
    actual_norms = calculate_norm(control_joints)
    
    # Calculate ideal norms
    ideal_norms = []
    start_config = np.array(goals[0])
    for goal_idx in range(1, len(goals)):
        start = np.array(goals[goal_idx - 1])
        goal = np.array(goals[goal_idx])
        num_points = len(control_joints[goal_column == (goal_idx - 1)])
        # print(num_points)
        interpolated_norms = interpolate_norms_continuous(start, goal, num_points)
        ideal_norms.extend(interpolated_norms)
    ideal_norms = np.array(ideal_norms)
    
    # Ensure both paths start from the same configuration
    start_norm = np.linalg.norm(start_config)
    actual_norms[0] = start_norm
    ideal_norms[0] = start_norm
    
    # Extract start and goal points for both paths
    ideal_goal_points = [start_config] + goals[1:]
    actual_goal_points = [start_config]
    for goal_idx in range(1, len(goals)):
        last_row = control_joints[goal_column == (goal_idx - 1)][-1]
        actual_goal_points.append(last_row)
    ideal_goal_points = np.array(ideal_goal_points)
    actual_goal_points = np.array(actual_goal_points)
    
    # Plot actual and ideal norms
    plt.plot(
        range(len(actual_norms)),
        actual_norms,
        linestyle='--',
        linewidth=4,
        color=color,
        alpha=0.9,
        label=f"{method} - Control"
    )
    plt.plot(
        range(len(ideal_norms)),
        ideal_norms,
        linestyle='-',
        linewidth=4,
        color=color,
        label=f"{method} - Plan"
    )
    
    # Highlight start and goal points and add dashed lines for distances
    for g_idx, (ideal_point, actual_point) in enumerate(zip(ideal_goal_points, actual_goal_points)):
        ideal_norm = np.linalg.norm(ideal_point)
        actual_norm = np.linalg.norm(actual_point)
        x_position = sum(len(control_joints[goal_column == (i - 1)]) for i in range(1, g_idx + 1)) - 1
        # plt.scatter(x_position, ideal_norm, color=color, marker='o', s=150)
        # plt.scatter(x_position, actual_norm, color=color, marker='o', s=150)

        if g_idx == 0:  # Start configuration
            plt.scatter(x_position, ideal_norm, color='#34C742', marker='o', s=150, label="Start Configuration" if method=='Image Space' else '')
            plt.scatter(x_position, actual_norm, color='#34C742', marker='o', s=150)
        elif g_idx == len(ideal_goal_points) - 1:  # Final goal configuration
            plt.scatter(x_position, ideal_norm, color='#CB48EB', marker='o', s=150, label="Final Goal Configuration" if method=='Image Space' else '')
            plt.scatter(x_position, actual_norm, color='#CB48EB', marker='o', s=150)
        else:  # Intermediate configurations
            plt.scatter(x_position, ideal_norm, color=color, marker='o', s=150, label="")
            plt.scatter(x_position, actual_norm, color=color, marker='o', s=150, label="Intermediate Goal Configurations in Path" if g_idx == 1 else "")

        plt.plot([x_position, x_position], [ideal_norm, actual_norm], linestyle='--', color='gray')
        distance = np.abs(ideal_norm - actual_norm)
        plt.text(x_position + 0.5, (ideal_norm + actual_norm) / 2, f"{distance:.2f}", fontsize=14, ha='left')

# Add legend, labels, and title
plt.title("Norm-Based Continuous Comparison for All Paths")
plt.xlabel("Total Number of Control Points")
plt.ylabel("Norm of Configurations")
# plt.legend(
#     # fontsize=28,
#     loc='upper right',
#     frameon=True,
#     fancybox=True,
#     shadow=True,
#     # title_fontsize=28,
#     edgecolor='black',
#     labelspacing=0.8,
#     prop={'size':14, 'weight': 'bold'},  # Make the legend text bold
# )
# plt.grid()
save_path = os.path.join(save_directory, "jt_traj_comp_01.svg")
plt.savefig(save_path)
plt.show()

# separate figure
for idx, (data, goals, color, label) in enumerate(zip(control_data, goal_configurations, colors, labels)):
    plt.figure(figsize=(20, 15))

    # Extract goal column and joint angles
    goal_column = data.iloc[:, 0].values
    control_joints = data.iloc[:, 1:4].to_numpy()
    
    # Calculate actual norms
    actual_norms = calculate_norm(control_joints)
    
    # Calculate ideal norms
    ideal_norms = []
    start_config = np.array(goals[0])
    for goal_idx in range(1, len(goals)):
        start = np.array(goals[goal_idx - 1])
        goal = np.array(goals[goal_idx])
        num_points = len(control_joints[goal_column == (goal_idx - 1)])
        interpolated_norms = interpolate_norms_continuous(start, goal, num_points)
        ideal_norms.extend(interpolated_norms)
    ideal_norms = np.array(ideal_norms)
    
    # Ensure both paths start from the same configuration
    start_norm = np.linalg.norm(start_config)
    actual_norms[0] = start_norm
    ideal_norms[0] = start_norm

    # Plot actual and ideal norms
    plt.plot(
        range(len(actual_norms)),
        actual_norms,
        linestyle='--',
        linewidth=4,
        color=color,
        alpha=0.7,
        label=f"{label} - Control"
    )
    plt.plot(
        range(len(ideal_norms)),
        ideal_norms,
        linestyle='-',
        linewidth=4,
        color=color,
        label=f"{label} - Plan"
    )

    # Highlight start and goal points
    ideal_goal_points = [start_config] + goals[1:]
    actual_goal_points = [start_config]
    for goal_idx in range(1, len(goals)):
        last_row = control_joints[goal_column == (goal_idx - 1)][-1]
        actual_goal_points.append(last_row)
    ideal_goal_points = np.array(ideal_goal_points)
    actual_goal_points = np.array(actual_goal_points)

    for g_idx, (ideal_point, actual_point) in enumerate(zip(ideal_goal_points, actual_goal_points)):
        ideal_norm = np.linalg.norm(ideal_point)
        actual_norm = np.linalg.norm(actual_point)
        x_position = sum(len(control_joints[goal_column == (i - 1)]) for i in range(1, g_idx + 1)) - 1
        # Plot the first configuration in green
        marker_color = 'green' if g_idx == 0 else color

        if g_idx == 0:  # Start configuration
            plt.scatter(x_position, ideal_norm, color='#34C742', marker='o', s=150, label="Start Configuration")
            plt.scatter(x_position, actual_norm, color='#34C742', marker='o', s=150)
        elif g_idx == len(ideal_goal_points) - 1:  # Final goal configuration
            plt.scatter(x_position, ideal_norm, color='#CB48EB', marker='o', s=150, label="Final Goal Configuration")
            plt.scatter(x_position, actual_norm, color='#CB48EB', marker='o', s=150)
        else:  # Intermediate configurations
            plt.scatter(x_position, ideal_norm, color=marker_color, marker='o', s=150, label="")
            plt.scatter(x_position, actual_norm, color=marker_color, marker='o', s=150, label="Intermediate Goal Configurations in Path" if g_idx == 1 else "")
        
        # # # Plot the start and goal points
        # plt.scatter(x_position, ideal_norm, color=color, marker='o', s=150, label="")
        # plt.scatter(x_position, actual_norm, color=color, marker='o', s=150, label="Configuration in Path" if g_idx == 1 else "")

        # Draw dashed lines between the corresponding start/goal points
        plt.plot([x_position, x_position], [ideal_norm, actual_norm], linestyle='--', color='gray')

        # Annotate the distance
        distance = np.abs(ideal_norm - actual_norm)
        plt.text(
            x_position + 0.5,  # Slight offset to avoid overlap
            (ideal_norm + actual_norm) / 2,  # Midpoint of the line
            f"{distance:.2f}",
            fontsize=14,
            ha='left'
        )

    # Add title, labels, legend, and grid
    plt.title(f"Norm-Based Continuous Comparison: {label}")
    plt.xlabel("Total Number of Control Points")
    plt.ylabel("Norm of Configurations")
    # plt.xlim(-100, 3500)
    # plt.ylim(1.5, 3.5)
#     plt.legend(
#     fontsize=28,
#     loc='upper right',
#     frameon=True,
#     fancybox=True,
#     shadow=True,
#     title_fontsize=28,
#     edgecolor='black',
#     labelspacing=1.2,
#     prop={'size':18, 'weight': 'bold'},  # Make the legend text bold
# )
    # plt.grid()
    save_path = os.path.join(save_directory, f"jt_{label.replace(' ', '_').lower()}_traj_01.svg")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure saved as {save_path}")
    plt.show()

# # Create a 3D plot
# fig = plt.figure(figsize=(20, 15))
# ax = fig.add_subplot(111, projection='3d')

# # Process each type of data
# for idx, (data, goals, color, label) in enumerate(zip(control_data, goal_configurations, colors, labels)):
#     # Extract goal column and joint angles
#     goal_column = data.iloc[:, 0].values
#     control_joints = data.iloc[:, 1:4].to_numpy()

#     # Calculate ideal trajectories
#     ideal_joints = []
#     for goal_idx in range(1, len(goals)):
#         start = np.array(goals[goal_idx - 1])
#         goal = np.array(goals[goal_idx])
#         num_points = len(control_joints[goal_column == (goal_idx - 1)])
#         interpolated_trajectory = np.linspace(start, goal, num_points)
#         ideal_joints.extend(interpolated_trajectory)
#     ideal_joints = np.array(ideal_joints)

#     # Extract goal points for highlighting
#     ideal_goal_points = [np.array(goals[0])] + [np.array(goal) for goal in goals[1:]]
#     actual_goal_points = [np.array(goals[0])]
#     for goal_idx in range(1, len(goals)):
#         last_row = control_joints[goal_column == (goal_idx - 1)][-1]
#         actual_goal_points.append(last_row)

#     # Convert to numpy arrays
#     ideal_goal_points = np.array(ideal_goal_points)
#     actual_goal_points = np.array(actual_goal_points)

#     # Plot actual joint trajectories
#     ax.plot(
#         control_joints[:, 0],
#         control_joints[:, 1],
#         control_joints[:, 2],
#         linestyle='--',
#         linewidth=2,
#         color=color,
#         alpha=0.8,
#         label=f"{label} - Actual"
#     )

#     # Plot ideal joint trajectories
#     ax.plot(
#         ideal_joints[:, 0],
#         ideal_joints[:, 1],
#         ideal_joints[:, 2],
#         linestyle='-',
#         linewidth=2,
#         color=color,
#         label=f"{label} - Ideal"
#     )

#     # Highlight goal points
#     for g_idx, (ideal_point, actual_point) in enumerate(zip(ideal_goal_points, actual_goal_points)):
#         # Highlight ideal and actual goal points
#         ax.scatter(
#             ideal_point[0], ideal_point[1], ideal_point[2],
#             color=color, marker='o', s=150,
#             label="Ideal Goals" if g_idx == 0 and idx == 0 else ''
#         )
#         ax.scatter(
#             actual_point[0], actual_point[1], actual_point[2],
#             color=color, marker='o', s=150,
#             label="Actual Goals" if g_idx == 0 and idx == 0 else ''
#         )

# # Set 3D plot labels and title
# ax.set_title("3D Joint Angle Trajectories (Ideal vs. Actual)", fontsize=18, weight='bold')
# ax.set_xlabel("Joint 1 Angle", fontsize=14, weight='bold')
# ax.set_ylabel("Joint 2 Angle", fontsize=14, weight='bold')
# ax.set_zlabel("Joint 3 Angle", fontsize=14, weight='bold')
# ax.legend(loc='upper right', fontsize=12)
# plt.show()

# for idx, (data, goals, color, label) in enumerate(zip(control_data, goal_configurations, colors, labels)):
#     # Create a 3D plot for each roadmap
#     fig = plt.figure(figsize=(20, 15))
#     ax = fig.add_subplot(111, projection='3d')

#     # Extract goal column and joint angles
#     goal_column = data.iloc[:, 0].values
#     control_joints = data.iloc[:, 1:4].to_numpy()

#     # Calculate ideal trajectories
#     ideal_joints = []
#     for goal_idx in range(1, len(goals)):
#         start = np.array(goals[goal_idx - 1])
#         goal = np.array(goals[goal_idx])
#         num_points = len(control_joints[goal_column == (goal_idx - 1)])
#         interpolated_trajectory = np.linspace(start, goal, num_points)
#         ideal_joints.extend(interpolated_trajectory)
#     ideal_joints = np.array(ideal_joints)

#     # Highlight goal transitions in the actual data
#     for goal_idx in range(1, len(goals)):
#         goal_mask = goal_column == (goal_idx - 1)
#         transition_data = control_joints[goal_mask]

#         if len(transition_data) > 0:
#             ax.plot(
#                 transition_data[:, 0],
#                 transition_data[:, 1],
#                 transition_data[:, 2],
#                 linestyle='--',
#                 linewidth=2,
#                 color=color,
#                 alpha=0.8
#             )

#     # Extract goal points for highlighting
#     ideal_goal_points = [np.array(goals[0])] + [np.array(goal) for goal in goals[1:]]
#     actual_goal_points = [np.array(goals[0])]
#     for goal_idx in range(1, len(goals)):
#         last_row = control_joints[goal_column == (goal_idx - 1)][-1]
#         actual_goal_points.append(last_row)

#     # Convert to numpy arrays
#     ideal_goal_points = np.array(ideal_goal_points)
#     actual_goal_points = np.array(actual_goal_points)

#     # Plot actual joint trajectories
#     ax.plot(
#         control_joints[:, 0],
#         control_joints[:, 1],
#         control_joints[:, 2],
#         linestyle='--',
#         linewidth=2,
#         color=color,
#         alpha=0.8,
#         label="Actual Trajectory"
#     )

#     # Plot ideal joint trajectories
#     ax.plot(
#         ideal_joints[:, 0],
#         ideal_joints[:, 1],
#         ideal_joints[:, 2],
#         linestyle='-',
#         linewidth=2,
#         color=color,
#         label="Ideal Trajectory"
#     )

#     # Highlight goal points
#     for g_idx, (ideal_point, actual_point) in enumerate(zip(ideal_goal_points, actual_goal_points)):
#         # Highlight ideal and actual goal points
#         ax.scatter(
#             ideal_point[0], ideal_point[1], ideal_point[2],
#             color=color, marker='o', s=150,
#             label="Ideal Goals" if g_idx == 0 else ''
#         )
#         ax.scatter(
#             actual_point[0], actual_point[1], actual_point[2],
#             color=color, marker='o', s=150,
#             label="Actual Goals" if g_idx == 0 else ''
#         )

#     # Set 3D plot labels and title
#     ax.set_title(f"3D Joint Angle Trajectories (Ideal vs. Actual) - {label}", fontsize=18, weight='bold')
#     ax.set_xlabel("Joint 1 Angle", fontsize=14, weight='bold')
#     ax.set_ylabel("Joint 2 Angle", fontsize=14, weight='bold')
#     ax.set_zlabel("Joint 3 Angle", fontsize=14, weight='bold')
#     ax.legend(loc='upper left', fontsize=12)
#     plt.show()


def calculate_perpendicular_distance(point, line_start, line_end):
    """
    Calculates the perpendicular distance from a point to a line segment defined by two points.
    """
    # Vector representation of the line segment
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    # Projection of point_vec onto line_vec
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(point_vec)  # Line segment is a point
    line_unit_vec = line_vec / line_len
    proj_length = np.dot(point_vec, line_unit_vec)
    
    # Clamp projection length to line segment length
    proj_length_clamped = max(0, min(proj_length, line_len))
    proj_point = line_start + proj_length_clamped * line_unit_vec
    
    # Perpendicular distance
    return np.linalg.norm(point - proj_point)
# Add deviation calculation for every point
deviation_stats = []

# Loop through each roadmap
for idx, (data, goals, color, label) in enumerate(zip(control_data, goal_configurations, colors, labels)):
    plt.figure(figsize=(20, 15))

    # Extract goal column and joint angles
    goal_column = data.iloc[:, 0].values
    control_joints = data.iloc[:, 1:4].to_numpy()

    # Interpolate ideal trajectories
    interpolated_ideal_trajectory = []
    for goal_idx in range(1, len(goals)):
        start = np.array(goals[goal_idx - 1])
        end = np.array(goals[goal_idx])
        num_points = len(control_joints[goal_column == (goal_idx - 1)])
        interpolated_segment = np.linspace(start, end, num_points)
        interpolated_ideal_trajectory.extend(interpolated_segment)
    interpolated_ideal_trajectory = np.array(interpolated_ideal_trajectory)

    # Calculate deviations
    deviations = []
    for point_idx in range(len(control_joints)):
        actual_point = control_joints[point_idx]
        ideal_point = interpolated_ideal_trajectory[point_idx]
        distance = np.linalg.norm(actual_point - ideal_point)
        deviations.append(distance)

    # Compute total points, total deviation, and maximum deviation
    # print(deviations[0], deviations[100], deviations[500], deviations[-1])
    total_points = len(deviations)
    total_deviation = sum(deviations)
    # mean_deviations = np.mean(deviations)
    max_deviation = max(deviations)
    deviation_stats.append({
        "Roadmap": label,
        "Total Points": total_points,
        "Total Deviation": total_deviation,
        # "Average Deviation": mean_deviations,
        "Max Deviation": max_deviation
    })

    # Visualization (optional)
    for point_idx in range(0, len(control_joints), 100):
        actual_point = control_joints[point_idx]
        ideal_point = interpolated_ideal_trajectory[point_idx]
        distance = np.linalg.norm(actual_point - ideal_point)

        # Plot the line connecting actual and interpolated points
        plt.plot(
            [actual_point[0], ideal_point[0]],
            [actual_point[1], ideal_point[1]],
            linestyle='--',
            color='gray',
            alpha=0.7
        )

        # Annotate the distance value
        plt.text(
            (actual_point[0] + ideal_point[0]) / 2,
            (actual_point[1] + ideal_point[1]) / 2,
            f"{distance:.2f}",
            fontsize=10,
            ha='center',
            va='center'
        )

    # Highlight goal configurations
    for goal in goals:
        plt.scatter(
            goal[0], goal[1],
            color=color,
            marker='o',
            s=150,
            label="Goal Configurations" if goal_idx == 1 else ""
        )

    plt.title(f"Trajectory Deviation Visualization: {label}")
    plt.xlabel("Joint X")
    plt.ylabel("Joint Y")
    plt.legend(loc='upper right')
    plt.grid()
    # plt.show()

# Print total points, total and max deviations for all roadmaps
import pandas as pd
deviation_stats_df = pd.DataFrame(deviation_stats)
print(deviation_stats_df)