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

# save_directory = "/media/jc-merlab/Crucial X9/paper_data/trajectory_pics/"

# # File paths
# file_paths = [
#     '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/ground_truth/astar_latest_no_obs/20/joint_angles.csv',
#     '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_no_obs/20/joint_angles.csv',
#     '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/euclidean/astar_latest_no_obs/1_a/joint_angles.csv'
# ]

# kp_file_paths = [
#     '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/ground_truth/astar_latest_no_obs/20/cp.csv',
#     '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_no_obs/20/cp.csv',
#     '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/euclidean/astar_latest_no_obs/1_a/cp.csv'
# ]

# jt_file_paths = [
#     '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/no_obs/nn_25_astar_custom_old/20/save_distances.csv',
#     '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/no_obs/nn_25_astar_custom_old/20/save_distances.csv',
#     '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/no_obs/20/save_distances.csv'
# ]

# # Initialize the goal_configurations list
# goal_configurations = []

# # Loop through each file and process it
# for file_path in jt_file_paths:
#     # Read the CSV file
#     df = pd.read_csv(file_path)
    
#     # Extract the 'Joint 1', 'Joint 2', 'Joint 3' columns
#     joints = df[['Joint 1', 'Joint 2', 'Joint 3']].values.tolist()
    
#     # Append the joint values as a list to goal_configurations
#     goal_configurations.append(joints)

# print("Ideal goal configurations", goal_configurations)

# initial_skip = 50

# def get_first_goal_row_count(kp_file, initial_skip):
#     """Get the row count for the first goal after skipping the header and initial rows."""
#     kp_df = pd.read_csv(kp_file, skiprows=initial_skip, header=0)
#     goal_columns = kp_df.iloc[:, 0].to_numpy()
#     first_goal = goal_columns[0]
#     kp_rows = (goal_columns == first_goal).sum()
#     print(f"First goal rows in {kp_file}: {kp_rows} (after skipping {initial_skip} rows).")
#     return kp_rows

# def get_last_goal_row_count(kp_file, initial_skip):
#     """Get the row count for the last goal after skipping rows."""
#     kp_df = pd.read_csv(kp_file, skiprows=initial_skip, header=0)
#     goal_columns = kp_df.iloc[:, 0].to_numpy()
#     unique_goals = np.unique(goal_columns)
#     last_goal = unique_goals[-1]
#     last_goal_rows = (goal_columns == last_goal).sum()
#     print(f"Last goal rows in {kp_file}: {last_goal_rows} (after skipping {initial_skip} rows).")
#     return last_goal_rows

# def get_total_goal_rows(control_file):
#     """Get the total rows for the first and last goals in control_file."""
#     control_df = pd.read_csv(control_file, header=None)
#     goal_columns = control_df.iloc[:, 0].to_numpy()
#     first_goal = goal_columns[0]
#     last_goal = goal_columns[-1]
#     total_first_goal = (goal_columns == first_goal).sum()
#     total_last_goal = (goal_columns == last_goal).sum()
#     return total_first_goal, total_last_goal

# def find_closest_to_ideal_last_goal(control_joints, ideal_last_goal_norm):
#     """Find the index in control_joints with the closest norm to the ideal last goal norm."""
#     norms = np.linalg.norm(control_joints, axis=1)
#     closest_index = np.argmin(np.abs(norms - ideal_last_goal_norm))
#     closest_value = control_joints[closest_index]  # Joint values at the closest index

#     print(f"Closest index to ideal last goal norm ({ideal_last_goal_norm:.3f}): {closest_index}")
#     print(f"Closest last goal value: {closest_value}")
#     return closest_index

# def read_control_data_limited(file_path, kp_first_rows, kp_last_rows, total_first_goal, total_last_goal, ideal_last_goal):
#     """Read control data with adjusted rows for first goal and closest last goal."""
#     # Adjust first goal
#     skip_first = total_first_goal - (kp_first_rows * 3)
#     print(f"Skipping {skip_first} rows for first goal in {file_path}")

#     control_df = pd.read_csv(file_path, skiprows=skip_first, header=None)

#     # Limit rows for the last goal
#     goal_columns = control_df.iloc[:, 0].to_numpy()
#     last_goal = goal_columns[-1]
#     last_goal_indices = np.where(goal_columns == last_goal)[0]
#     limit_last_rows = kp_last_rows * 3
#     last_goal_indices_limited = last_goal_indices[:limit_last_rows]

#     print(f"Limiting last goal rows to {limit_last_rows} in {file_path}")

#     # Find the closest index to the ideal last goal norm
#     last_goal_joints = control_df.iloc[last_goal_indices_limited, 1:4].to_numpy()
#     ideal_last_goal_norm = np.linalg.norm(ideal_last_goal)
#     closest_index = find_closest_to_ideal_last_goal(last_goal_joints, ideal_last_goal_norm)

#     # Combine earlier rows and the closest last goal row
#     rows_to_keep = np.concatenate([np.where(goal_columns != last_goal)[0], last_goal_indices_limited[:closest_index + 1]])
#     control_df_limited = control_df.iloc[rows_to_keep]

#     print(f"Final control data rows: {len(control_df_limited)}")
#     return control_df_limited

# # Process Each File Pair
# control_data = []

# for idx, (kp_file, file_path) in enumerate(zip(kp_file_paths, file_paths)):
#     print(f"\nProcessing: {kp_file} and {file_path}")
    
#     # Step 1: Get first and last goal row counts
#     kp_first_rows = get_first_goal_row_count(kp_file, initial_skip=50)
#     kp_last_rows = get_last_goal_row_count(kp_file, initial_skip=50)

#     # Step 2: Get total rows for first and last goals
#     total_first_goal, total_last_goal = get_total_goal_rows(file_path)

#     # Step 3: Use the last goal from goal_configurations
#     ideal_last_goal = goal_configurations[idx][-1]

#     # Step 4: Read control data
#     control_df = read_control_data_limited(
#         file_path, kp_first_rows, kp_last_rows, total_first_goal, total_last_goal, ideal_last_goal
#     )
#     control_data.append(control_df)

# print("\nFinished processing all files.")

# # Load the control data
# # control_data = [pd.read_csv(fp, header=758) for fp in file_paths]
# # control_data = [read_control_data(fp) for fp in file_paths]

# # Helper functions
# def calculate_norm(configurations):
#     """Calculates the Euclidean norm of the joint angles for each configuration."""
#     return np.linalg.norm(configurations, axis=1)

# # Determine the maximum number of goal configurations across all cases
# global_x_max = max(len(goals) for goals in goal_configurations)

# # Pre-compute the maximum norm across all datasets for consistent y-axis limits
# global_y_max = max([np.max(calculate_norm(data.iloc[:, 1:4].to_numpy())) for data in control_data])


# def interpolate_norms_continuous(start_config, goal_config, num_points):
#     """Interpolates the norms of the start and goal configurations."""
#     start_norm = np.linalg.norm(start_config)
#     goal_norm = np.linalg.norm(goal_config)
#     return np.linspace(start_norm, goal_norm, num_points)

# # Colors and labels for plotting
# colors=['#40B0A6', '#5D3A9B', '#D41159']
# labels = ['Ground Truth', 'Learned', 'Image Space']

# # Define consistent x-positions for start and goal points
# goal_x_positions = [0]  # Start from 0
# for goals in goal_configurations:
#     goal_x_positions.extend(np.cumsum([len(goals) for _ in range(1, len(goals))]))

# # Plot setup
# plt.figure(figsize=(20, 15))

# # Process each file
# # for idx, (data, goals, color, method) in enumerate(zip(control_data, goal_configurations, colors, labels)):
# #     # Extract goal column and joint angles
# #     goal_column = data.iloc[:, 0].values
# #     control_joints = data.iloc[:, 1:4].to_numpy()
    
# #     # Calculate actual norms
# #     actual_norms = calculate_norm(control_joints)
    
# #     # Calculate ideal norms
# #     ideal_norms = []
# #     start_config = np.array(goals[0])
# #     for goal_idx in range(1, len(goals)):
# #         start = np.array(goals[goal_idx - 1])
# #         goal = np.array(goals[goal_idx])
# #         num_points = len(control_joints[goal_column == (goal_idx - 1)])
# #         # print(num_points)
# #         interpolated_norms = interpolate_norms_continuous(start, goal, num_points)
# #         ideal_norms.extend(interpolated_norms)
# #     ideal_norms = np.array(ideal_norms)
    
# #     # Ensure both paths start from the same configuration
# #     start_norm = np.linalg.norm(start_config)
# #     actual_norms[0] = start_norm
# #     ideal_norms[0] = start_norm
    
# #     # Extract start and goal points for both paths
# #     ideal_goal_points = [start_config] + goals[1:]
# #     actual_goal_points = [start_config]
# #     for goal_idx in range(1, len(goals)):
# #         last_row = control_joints[goal_column == (goal_idx - 1)][-1]
# #         actual_goal_points.append(last_row)
# #     ideal_goal_points = np.array(ideal_goal_points)
# #     actual_goal_points = np.array(actual_goal_points)
    
# #     # Plot actual and ideal norms
# #     plt.plot(
# #         range(len(actual_norms)),
# #         actual_norms,
# #         linestyle='--',
# #         linewidth=4,
# #         color=color,
# #         alpha=0.9,
# #         label=f"{method} - Control"
# #     )
# #     plt.plot(
# #         range(len(ideal_norms)),
# #         ideal_norms,
# #         linestyle='-',
# #         linewidth=4,
# #         color=color,
# #         label=f"{method} - Plan"
# #     )

   
# #     # Highlight start and goal points and add dashed lines for distances
# #     for g_idx, (ideal_point, actual_point) in enumerate(zip(ideal_goal_points, actual_goal_points)):
# #         ideal_norm = np.linalg.norm(ideal_point)
# #         actual_norm = np.linalg.norm(actual_point)
# #         x_position = sum(len(control_joints[goal_column == (i - 1)]) for i in range(1, g_idx + 1)) - 1
# #         # plt.scatter(x_position, ideal_norm, color=color, marker='o', s=150)
# #         # plt.scatter(x_position, actual_norm, color=color, marker='o', s=150)

# #         if g_idx == 0:  # Start configuration
# #             plt.scatter(x_position, ideal_norm, color='#34C742', marker='o', s=150, label="Start Configuration" if method=='Image Space' else '')
# #             plt.scatter(x_position, actual_norm, color='#34C742', marker='o', s=150)
# #         elif g_idx == len(ideal_goal_points) - 1:  # Final goal configuration
# #             plt.scatter(x_position, ideal_norm, color='#CB48EB', marker='o', s=150, label="Final Goal Configuration" if method=='Image Space' else '')
# #             plt.scatter(x_position, actual_norm, color='#CB48EB', marker='o', s=150)
# #         else:  # Intermediate configurations
# #             plt.scatter(x_position, ideal_norm, color=color, marker='o', s=150, label="")
# #             plt.scatter(x_position, actual_norm, color=color, marker='o', s=150, label="Intermediate Goal Configurations in Path" if g_idx == 1 else "")

# #         plt.plot([x_position, x_position], [ideal_norm, actual_norm], linestyle='--', color='gray')
# #         distance = np.abs(ideal_norm - actual_norm)
# #         plt.text(x_position + 0.5, (ideal_norm + actual_norm) / 2, f"{distance:.2f}", fontsize=14, ha='left')

# # # Add legend, labels, and title
# # # plt.title("Norm-Based Continuous Comparison for All Paths")
# # # plt.xlabel("Total Number of Control Points")
# # # plt.ylabel("Norm of Configurations")
# # plt.xlim()
# # # plt.legend(
# # #     # fontsize=28,
# # #     loc='upper right',
# # #     frameon=True,
# # #     fancybox=True,
# # #     shadow=True,
# # #     # title_fontsize=28,
# # #     edgecolor='black',
# # #     labelspacing=0.8,
# # #     prop={'size':14, 'weight': 'bold'},  # Make the legend text bold
# # # )
# # # plt.grid()
# # save_path = os.path.join(save_directory, "jt_traj_comp_09a.svg")
# # plt.savefig(save_path)
# # plt.show()

# # # for idx, (data, goals, color, method) in enumerate(zip(control_data, goal_configurations, colors, labels)):
# # #     # Extract goal column and joint angles
# # #     goal_column = data.iloc[:, 0].values
# # #     control_joints = data.iloc[:, 1:4].to_numpy()
    
# # #     # Stop plotting at the final goal
# # #     final_goal_index = np.where(goal_column == (len(goals) - 2))[0][-1]  # Last index for the final goal
# # #     control_joints = control_joints[:final_goal_index + 1]
# # #     goal_column = goal_column[:final_goal_index + 1]

# # #     # Calculate actual norms
# # #     actual_norms = calculate_norm(control_joints)
    
# # #     # Calculate ideal norms
# # #     ideal_norms = []
# # #     start_config = np.array(goals[0])
# # #     for goal_idx in range(1, len(goals)):
# # #         start = np.array(goals[goal_idx - 1])
# # #         goal = np.array(goals[goal_idx])
# # #         num_points = len(control_joints[goal_column == (goal_idx - 1)])
# # #         interpolated_norms = interpolate_norms_continuous(start, goal, num_points)
# # #         ideal_norms.extend(interpolated_norms)
# # #     ideal_norms = np.array(ideal_norms[:len(actual_norms)])  # Trim ideal norms to match actual

# # #     # Ensure both paths start from the same configuration
# # #     start_norm = np.linalg.norm(start_config)
# # #     actual_norms[0] = start_norm
# # #     ideal_norms[0] = start_norm

# # #     # Plot actual and ideal norms
# # #     plt.plot(
# # #         range(len(actual_norms)),
# # #         actual_norms,
# # #         linestyle='--',
# # #         linewidth=4,
# # #         color=color,
# # #         alpha=0.9,
# # #         label=f"{method} - Control"
# # #     )
# # #     plt.plot(
# # #         range(len(ideal_norms)),
# # #         ideal_norms,
# # #         linestyle='-',
# # #         linewidth=4,
# # #         color=color,
# # #         label=f"{method} - Plan"
# # #     )

# # #     # Highlight start and goal points
# # #     plt.scatter(0, start_norm, color='#34C742', marker='o', s=150, label="Start Configuration")
# # #     plt.scatter(len(actual_norms) - 1, np.linalg.norm(goals[-1]), color='#CB48EB', marker='o', s=150, label="Final Goal Configuration")

# # #     # Add dashed lines and distances
# # #     for g_idx, goal in enumerate(goals[1:]):
# # #         x_position = sum(len(control_joints[goal_column == i]) for i in range(g_idx)) - 1
# # #         ideal_norm = np.linalg.norm(goal)
# # #         actual_norm = actual_norms[x_position]

# # #         plt.plot([x_position, x_position], [ideal_norm, actual_norm], linestyle='--', color='gray')
# # #         distance = np.abs(ideal_norm - actual_norm)
# # #         plt.text(x_position + 0.5, (ideal_norm + actual_norm) / 2, f"{distance:.2f}", fontsize=14, ha='left')

# # # # Finalize plot
# # # plt.title("Norm-Based Continuous Comparison for All Paths")
# # # plt.xlabel("Total Number of Control Points")
# # # plt.ylabel("Norm of Configurations")
# # # plt.legend(loc='upper right')
# # # save_path = os.path.join(save_directory, "jt_traj_comp_obs_obs_19_cleaned.svg")
# # # plt.savefig(save_path, bbox_inches='tight')
# # # plt.show()

# # # separate figure
# # for idx, (data, goals, color, label) in enumerate(zip(control_data, goal_configurations, colors, labels)):
# #     plt.figure(figsize=(20, 15))

# #     # Extract goal column and joint angles
# #     goal_column = data.iloc[:, 0].values
# #     control_joints = data.iloc[:, 1:4].to_numpy()
    
# #     # Calculate actual norms
# #     actual_norms = calculate_norm(control_joints)
    
# #     # Calculate ideal norms
# #     ideal_norms = []
# #     start_config = np.array(goals[0])
# #     for goal_idx in range(1, len(goals)):
# #         start = np.array(goals[goal_idx - 1])
# #         goal = np.array(goals[goal_idx])
# #         num_points = len(control_joints[goal_column == (goal_idx - 1)])
# #         interpolated_norms = interpolate_norms_continuous(start, goal, num_points)
# #         ideal_norms.extend(interpolated_norms)
# #     ideal_norms = np.array(ideal_norms)
    
# #     # Ensure both paths start from the same configuration
# #     start_norm = np.linalg.norm(start_config)
# #     actual_norms[0] = start_norm
# #     ideal_norms[0] = start_norm

# #     # Plot actual and ideal norms
# #     plt.plot(
# #         range(len(actual_norms)),
# #         actual_norms,
# #         linestyle='--',
# #         linewidth=5,
# #         color=color,
# #         alpha=0.7,
# #         label=f"{label} - Control Trajectory"
# #     )
# #     plt.plot(
# #         range(len(ideal_norms)),
# #         ideal_norms,
# #         linestyle='-',
# #         linewidth=5,
# #         color=color,
# #         label=f"{label} - Planned Path"
# #     )

# #     # Highlight start and goal points
# #     ideal_goal_points = [start_config] + goals[1:]
# #     actual_goal_points = [start_config]
# #     for goal_idx in range(1, len(goals)):
# #         last_row = control_joints[goal_column == (goal_idx - 1)][-1]
# #         actual_goal_points.append(last_row)
# #     ideal_goal_points = np.array(ideal_goal_points)
# #     actual_goal_points = np.array(actual_goal_points)

# #     for g_idx, (ideal_point, actual_point) in enumerate(zip(ideal_goal_points, actual_goal_points)):
# #         ideal_norm = np.linalg.norm(ideal_point)
# #         actual_norm = np.linalg.norm(actual_point)
# #         x_position = sum(len(control_joints[goal_column == (i - 1)]) for i in range(1, g_idx + 1)) - 1
# #         # Plot the first configuration in green
# #         marker_color = 'green' if g_idx == 0 else color

# #         if g_idx == 0:  # Start configuration
# #             plt.scatter(x_position, ideal_norm, color='#34C742', marker='o', s=150, label="Start Configuration")
# #             plt.scatter(x_position, actual_norm, color='#34C742', marker='o', s=150)
# #         elif g_idx == len(ideal_goal_points) - 1:  # Final goal configuration
# #             plt.scatter(x_position, ideal_norm, color='#CB48EB', marker='o', s=150, label="Final Goal Configuration")
# #             plt.scatter(x_position, actual_norm, color='#CB48EB', marker='o', s=150)
# #         else:  # Intermediate configurations
# #             plt.scatter(x_position, ideal_norm, color=marker_color, marker='o', s=150, label="")
# #             plt.scatter(x_position, actual_norm, color=marker_color, marker='o', s=150, label="Intermediate Goal Configurations in Path" if g_idx == 1 else "")
        
# #         # # # Plot the start and goal points
# #         # plt.scatter(x_position, ideal_norm, color=color, marker='o', s=150, label="")
# #         # plt.scatter(x_position, actual_norm, color=color, marker='o', s=150, label="Configuration in Path" if g_idx == 1 else "")

# #         # Draw dashed lines between the corresponding start/goal points
# #         plt.plot([x_position, x_position], [ideal_norm, actual_norm], linestyle='--', color='gray')

# #         # Annotate the distance
# #         distance = np.abs(ideal_norm - actual_norm)
# #         plt.text(
# #             x_position + 0.5,  # Slight offset to avoid overlap
# #             (ideal_norm + actual_norm) / 2,  # Midpoint of the line
# #             f"{distance:.2f}",
# #             fontsize=14,
# #             ha='left'
# #         )

# #     # Add title, labels, legend, and grid
# #     # plt.title(f"Norm-Based Continuous Comparison: {label}")
# #     # plt.xlabel("Total Number of Control Points")
# #     # plt.ylabel("Norm of Configurations")
# #     plt.xlim(-100, 3200)
# #     plt.ylim(1.5, 3.5)
# #     plt.legend(
# #     fontsize=24,
# #     loc='upper right',
# #     frameon=True,
# #     fancybox=True,
# #     shadow=True,
# #     title_fontsize=24,
# #     edgecolor='black',
# #     labelspacing=1.2,
# #     prop={'size':16, 'weight': 'bold'},  # Make the legend text bold
# # )
# #     # plt.grid()
# #     # plt.tick_params(axis='both', which='major', labelsize=18)
# #     for label in plt.gca().get_xticklabels():
# #         label.set_fontweight('bold')
# #         label.set_fontsize(20)
# #     for label in plt.gca().get_yticklabels():
# #         label.set_fontweight('bold')
# #         label.set_fontsize(20)


# #     save_path = os.path.join(save_directory, f"jt_{label.get_text().replace(' ', '_').lower()}_traj_09a.svg")
# #     plt.savefig(save_path, bbox_inches='tight')
# #     print(f"Figure saved as {save_path}")
# #     plt.show()


# def calculate_perpendicular_distance(point, line_start, line_end):
#     """
#     Calculates the perpendicular distance from a point to a line segment defined by two points.
#     """
#     # Vector representation of the line segment
#     line_vec = line_end - line_start
#     point_vec = point - line_start
    
#     # Projection of point_vec onto line_vec
#     line_len = np.linalg.norm(line_vec)
#     if line_len == 0:
#         return np.linalg.norm(point_vec)  # Line segment is a point
#     line_unit_vec = line_vec / line_len
#     proj_length = np.dot(point_vec, line_unit_vec)
    
#     # Clamp projection length to line segment length
#     proj_length_clamped = max(0, min(proj_length, line_len))
#     proj_point = line_start + proj_length_clamped * line_unit_vec
    
#     # Perpendicular distance
#     return np.linalg.norm(point - proj_point)
# # Add deviation calculation for every point
# # deviation_stats = []

# # # Loop through each roadmap
# # for idx, (data, goals, color, label) in enumerate(zip(control_data, goal_configurations, colors, labels)):
# #     plt.figure(figsize=(20, 15))

# #     # Extract goal column and joint angles
# #     goal_column = data.iloc[:, 0].values
# #     control_joints = data.iloc[:, 1:4].to_numpy()

# #     print(control_joints.shape[0])

# #     goal_column = goal_column[:control_joints.shape[0]]

# #     # Interpolate ideal trajectories
# #     interpolated_ideal_trajectory = []
# #     deviations = []
# #     for goal_idx in range(1, len(goals)):
# #         start = np.array(goals[goal_idx - 1])
# #         end = np.array(goals[goal_idx])
# #         # num_points = len(control_joints[goal_column == (goal_idx - 1)])
# #         goal_mask = goal_column == (goal_idx - 1)
# #         num_points = np.sum(goal_mask)
# #         if num_points == 0:
# #             raise ValueError(f"No rows found for goal {goal_idx - 1}.")
        
# #         interpolated_segment = np.linspace(start, end, num_points)
# #         interpolated_ideal_trajectory.extend(interpolated_segment)
# #         # Calculate deviations for each point

# #         actual_points = control_joints[goal_mask]

# #         for actual_point, interpolated_point in zip(actual_points, interpolated_segment):

# #             distance = np.linalg.norm(actual_point - interpolated_point)

# #             deviations.append(distance)

# #     interpolated_ideal_trajectory = np.array(interpolated_ideal_trajectory)

# #     # Calculate deviations
# #     # deviations = []
# #     # for point_idx in range(len(control_joints)):
# #     #     actual_point = control_joints[point_idx]
# #     #     ideal_point = interpolated_ideal_trajectory[point_idx]
# #     #     distance = np.linalg.norm(actual_point - ideal_point)
# #     #     deviations.append(distance)

# #     # Compute total points, total deviation, and maximum deviation
# #     # print(deviations[0], deviations[100], deviations[500], deviations[-1])
# #     total_points = len(deviations)
# #     total_deviation = sum(deviations)
# #     # mean_deviations = np.mean(deviations)
# #     max_deviation = max(deviations)
# #     deviation_stats.append({
# #         "Roadmap": label,
# #         "Total Points": total_points,
# #         "Total Deviation": total_deviation,
# #         # "Average Deviation": mean_deviations,
# #         "Max Deviation": max_deviation
# #     })

# # deviation_stats_df = pd.DataFrame(deviation_stats)
# # print(deviation_stats_df)

# deviation_stats = []

# for idx, (data, goals, color, label) in enumerate(zip(control_data, goal_configurations, colors, labels)):
#     plt.figure(figsize=(20, 15))

#     # Extract goal column and joint angles
#     goal_column = data.iloc[:, 0].values
#     control_joints = data.iloc[:, 1:4].to_numpy()

#     # Plot only the recorded control trajectory
#     control_length = len(control_joints)

#     # Interpolate the entire ideal trajectory
#     interpolated_ideal_trajectory = []
#     deviations = []
#     for goal_idx in range(1, len(goals)):
#         start = np.array(goals[goal_idx - 1])
#         end = np.array(goals[goal_idx])
#         num_points = len(control_joints[goal_column == (goal_idx - 1)])
#         interpolated_segment = np.linspace(start, end, num_points)
#         interpolated_ideal_trajectory.extend(interpolated_segment)

#     interpolated_ideal_trajectory = np.array(interpolated_ideal_trajectory)

#     # Ensure ideal trajectory is plotted fully
#     ideal_norms = np.linalg.norm(interpolated_ideal_trajectory, axis=1)
#     plt.plot(range(len(ideal_norms)), ideal_norms, linestyle='-', linewidth=4, color=color, label=f"{label} - Plan")

#     # Plot the control trajectory
#     actual_norms = np.linalg.norm(control_joints, axis=1)
#     plt.plot(range(control_length), actual_norms, linestyle='--', linewidth=4, color=color, alpha=0.7, label=f"{label} - Control")

#     # Calculate deviations only for the recorded control trajectory
#     interpolated_trajectory_for_deviation = interpolated_ideal_trajectory[:control_length]
#     for actual_point, ideal_point in zip(control_joints, interpolated_trajectory_for_deviation):
#         distance = np.linalg.norm(actual_point - ideal_point)
#         deviations.append(distance)

#     # Compute deviation stats
#     total_points = len(deviations)
#     total_deviation = sum(deviations)
#     max_deviation = max(deviations)
#     deviation_stats.append({
#         "Roadmap": label,
#         "Total Points": total_points,
#         "Total Deviation": total_deviation,
#         "Max Deviation": max_deviation
#     })

#     # Highlight start and final goal points
#     plt.scatter(0, actual_norms[0], color='#34C742', marker='o', s=150, label="Start Configuration")
#     plt.scatter(control_length - 1, actual_norms[-1], color='#CB48EB', marker='o', s=150, label="Final Goal Configuration")

#     # Finalize plot
#     plt.title(f"Norm-Based Continuous Comparison: {label}")
#     plt.xlabel("Total Number of Control Points")
#     plt.ylabel("Norm of Configurations")
#     plt.legend(fontsize=14, loc='upper right')
#     plt.grid()

#     save_path = os.path.join(save_directory, f"jt_{label.replace(' ', '_').lower()}_traj.svg")
#     # plt.savefig(save_path, bbox_inches='tight')
#     print(f"Figure saved as {save_path}")
#     # plt.show()

# # Save deviation stats as a DataFrame
# deviation_stats_df = pd.DataFrame(deviation_stats)
# print(deviation_stats_df)

# # Process each file
# for idx, (data, goals, color, method) in enumerate(zip(control_data, goal_configurations, colors, labels)):
#     # Extract goal column and joint angles
#     goal_column = data.iloc[:, 0].values
#     control_joints = data.iloc[:, 1:4].to_numpy()

#     # Calculate the number of actual and ideal goals
#     actual_goal_count = len(np.unique(goal_column))
#     ideal_goal_count = len(goals)

#     # Limit actual trajectory if fewer actual goals than ideal goals
#     if actual_goal_count < ideal_goal_count:
#         print(f"Fewer actual goals ({actual_goal_count}) than ideal goals ({ideal_goal_count}) for {method}.")
#         matched_goals = goals[:actual_goal_count]  # Use only matching goals
#     else:
#         matched_goals = goals

#     # Calculate actual norms
#     actual_norms = calculate_norm(control_joints)

#     # Calculate ideal norms for matched goals
#     ideal_norms = []
#     start_config = np.array(matched_goals[0])
#     for goal_idx in range(1, len(matched_goals)):
#         start = np.array(matched_goals[goal_idx - 1])
#         goal = np.array(matched_goals[goal_idx])
#         num_points = len(control_joints[goal_column == (goal_idx - 1)])
#         interpolated_norms = interpolate_norms_continuous(start, goal, num_points)
#         ideal_norms.extend(interpolated_norms)

#     # Add unmatched ideal goals if any
#     if actual_goal_count < ideal_goal_count:
#         for unmatched_goal in goals[actual_goal_count:]:
#             unmatched_norm = np.linalg.norm(np.array(unmatched_goal))
#             ideal_norms.append(unmatched_norm)

#     # Ensure both paths start from the same configuration
#     start_norm = np.linalg.norm(start_config)
#     actual_norms[0] = start_norm
#     ideal_norms[0] = start_norm

#     # Plot actual and ideal norms
#     plt.plot(
#         range(len(actual_norms)),
#         actual_norms,
#         linestyle='--',
#         linewidth=4,
#         color=color,
#         alpha=0.9,
#         label=f"{method} - Control"
#     )
#     plt.plot(
#         range(len(ideal_norms)),
#         ideal_norms,
#         linestyle='-',
#         linewidth=4,
#         color=color,
#         label=f"{method} - Plan"
#     )

#     # Highlight start and goal points for both ideal and actual
#     ideal_goal_points = [start_config] + matched_goals[1:]
#     actual_goal_points = [start_config]
#     for goal_idx in range(1, len(matched_goals)):
#         last_row = control_joints[goal_column == (goal_idx - 1)][-1]
#         actual_goal_points.append(last_row)

#     for g_idx, (ideal_point, actual_point) in enumerate(zip(ideal_goal_points, actual_goal_points)):
#         ideal_norm = np.linalg.norm(ideal_point)
#         actual_norm = np.linalg.norm(actual_point)
#         x_position = sum(len(control_joints[goal_column == (i - 1)]) for i in range(1, g_idx + 1)) - 1

#         if g_idx == 0:  # Start configuration
#             plt.scatter(x_position, ideal_norm, color='#34C742', marker='o', s=150, label="Start Configuration" if method == 'Image Space' else '')
#             plt.scatter(x_position, actual_norm, color='#34C742', marker='o', s=150)
#         elif g_idx == len(matched_goals) - 1:  # Final goal configuration
#             plt.scatter(x_position, ideal_norm, color='#CB48EB', marker='o', s=150, label="Final Goal Configuration" if method == 'Image Space' else '')
#             plt.scatter(x_position, actual_norm, color='#CB48EB', marker='o', s=150)
#         else:  # Intermediate configurations
#             plt.scatter(x_position, ideal_norm, color=color, marker='o', s=150, label="Intermediate Goal Configurations" if g_idx == 1 else "")
#             plt.scatter(x_position, actual_norm, color=color, marker='o', s=150)

#         plt.plot([x_position, x_position], [ideal_norm, actual_norm], linestyle='--', color='gray')
#         distance = np.abs(ideal_norm - actual_norm)
#         plt.text(x_position + 0.5, (ideal_norm + actual_norm) / 2, f"{distance:.2f}", fontsize=14, ha='left')

#     # Plot unmatched ideal goals (if any)
#     if actual_goal_count < ideal_goal_count:
#         unmatched_x_positions = list(range(len(actual_norms), len(ideal_norms)))
#         unmatched_ideal_norms = ideal_norms[len(actual_norms):]
#         plt.scatter(unmatched_x_positions, unmatched_ideal_norms, color=color, marker='o', s=150, alpha=0.5, label="Unmatched Ideal Goals")

# # Finalize and save the plot
# plt.title("Norm-Based Continuous Comparison for All Paths")
# plt.xlabel("Total Number of Control Points")
# plt.ylabel("Norm of Configurations")
# plt.legend(fontsize=14, loc='upper right')
# plt.grid()
# save_path = os.path.join(save_directory, "jt_traj_comp_09a.svg")
# # plt.savefig(save_path)
# # plt.show()

# for idx, (data, goals, color, label) in enumerate(zip(control_data, goal_configurations, colors, labels)):
#     plt.figure(figsize=(20, 15))

#     # Extract goal column and joint angles
#     goal_column = data.iloc[:, 0].values.astype(int)
#     control_joints = data.iloc[:, 1:4].to_numpy()

#     # Calculate actual norms
#     actual_norms = calculate_norm(control_joints)

#     # Determine the number of actual goals
#     actual_goal_indices = np.unique(goal_column)
#     actual_goal_count = len(actual_goal_indices)

#     # Adjust the number of goals to match the actual data
#     matched_goals = goals[:actual_goal_count + 1]  # +1 to include the start configuration

#     # Calculate ideal norms up to the number of actual goals
#     ideal_norms = []
#     start_config = np.array(matched_goals[0])
#     for goal_idx in range(1, len(matched_goals)):
#         start = np.array(matched_goals[goal_idx - 1])
#         goal = np.array(matched_goals[goal_idx])
#         num_points = len(control_joints[goal_column == (goal_idx - 1)])
#         if num_points == 0:
#             print(f"No control points for goal {goal_idx - 1}, skipping.")
#             continue
#         interpolated_norms = interpolate_norms_continuous(start, goal, num_points)
#         ideal_norms.extend(interpolated_norms)
#     ideal_norms = np.array(ideal_norms)

#     # Ensure both paths start from the same configuration
#     start_norm = np.linalg.norm(start_config)
#     actual_norms[0] = start_norm
#     if len(ideal_norms) > 0:
#         ideal_norms[0] = start_norm

#     # Plot actual and ideal norms
#     plt.plot(
#         range(len(actual_norms)),
#         actual_norms,
#         linestyle='--',
#         linewidth=5,
#         color=color,
#         alpha=0.7,
#         label=f"{label} - Control Trajectory"
#     )
#     plt.plot(
#         range(len(ideal_norms)),
#         ideal_norms,
#         linestyle='-',
#         linewidth=5,
#         color=color,
#         label=f"{label} - Planned Path"
#     )

#     # Highlight start and goal points
#     ideal_goal_points = matched_goals
#     actual_goal_points = [start_config]

#     for goal_idx in range(1, len(matched_goals)):
#         indices = np.where(goal_column == (goal_idx - 1))[0]
#         if len(indices) == 0:
#             print(f"No data for goal_idx {goal_idx - 1}, stopping.")
#             break
#         last_row = control_joints[indices][-1]
#         actual_goal_points.append(last_row)

#     ideal_goal_points = np.array(ideal_goal_points)
#     actual_goal_points = np.array(actual_goal_points)

#     for g_idx, (ideal_point, actual_point) in enumerate(zip(ideal_goal_points, actual_goal_points)):
#         ideal_norm = np.linalg.norm(ideal_point)
#         actual_norm = np.linalg.norm(actual_point)
#         x_position = sum(len(control_joints[goal_column == (i - 1)]) for i in range(1, g_idx + 1)) - 1

#         if g_idx == 0:  # Start configuration
#             plt.scatter(x_position, ideal_norm, color='#34C742', marker='o', s=150, label="Start Configuration" if idx == 0 else "")
#             plt.scatter(x_position, actual_norm, color='#34C742', marker='o', s=150)
#         elif g_idx == len(actual_goal_points) - 1:  # Last actual goal configuration
#             plt.scatter(x_position, ideal_norm, color='#CB48EB', marker='o', s=150, label="Final Goal Configuration" if idx == 0 else "")
#             plt.scatter(x_position, actual_norm, color='#CB48EB', marker='o', s=150)
#         else:  # Intermediate configurations
#             plt.scatter(x_position, ideal_norm, color=color, marker='o', s=150, label="Intermediate Goal Configurations" if g_idx == 1 and idx == 0 else "")
#             plt.scatter(x_position, actual_norm, color=color, marker='o', s=150)

#         # Draw dashed lines between the corresponding start/goal points
#         plt.plot([x_position, x_position], [ideal_norm, actual_norm], linestyle='--', color='gray')

#         # Annotate the distance
#         distance = np.abs(ideal_norm - actual_norm)
#         plt.text(
#             x_position + 0.5,
#             (ideal_norm + actual_norm) / 2,
#             f"{distance:.2f}",
#             fontsize=14,
#             ha='left'
#         )

#     # Handle unmatched ideal goals (if any)
#     if len(goals) > len(matched_goals):
#         remaining_goals = goals[len(matched_goals):]
#         # Plot the remaining ideal norms
#         remaining_ideal_norms = []
#         for goal_idx in range(len(matched_goals), len(goals)):
#             start = np.array(goals[goal_idx - 1])
#             goal = np.array(goals[goal_idx])
#             num_points = 10  # Use a fixed number of points for visualization
#             interpolated_norms = interpolate_norms_continuous(start, goal, num_points)
#             remaining_ideal_norms.extend(interpolated_norms)
#         remaining_ideal_norms = np.array([np.linalg.norm(x) for x in remaining_ideal_norms])

#         # Calculate start index for plotting
#         start_idx = len(actual_norms)

#         # Plot the remaining ideal norms
#         plt.plot(
#             range(start_idx, start_idx + len(remaining_ideal_norms)),
#             remaining_ideal_norms,
#             linestyle='-',
#             linewidth=5,
#             color=color,
#             alpha=0.5,
#             label=f"{label} - Planned Path (Unreached Goals)"
#         )

#         # Plot the unmatched ideal goal points
#         x_position = start_idx
#         for goal in remaining_goals:
#             goal_norm = np.linalg.norm(goal)
#             plt.scatter(x_position, goal_norm, color=color, marker='o', s=150, alpha=0.5)
#             x_position += 10  # Increment by the number of points used in interpolation

#     # Finalize the plot
#     plt.title(f"Norm-Based Continuous Comparison: {label}")
#     plt.xlabel("Total Number of Control Points")
#     plt.ylabel("Norm of Configurations")
#     plt.legend(fontsize=14, loc='upper right')
#     plt.grid()
#     save_path = os.path.join(save_directory, f"jt_{label.replace(' ', '_').lower()}_traj.svg")
#     # plt.savefig(save_path, bbox_inches='tight')
#     print(f"Figure saved as {save_path}")
#     # plt.show()

# interpolated_ideal_trajectory = []
# deviations = []
# for goal_idx in range(1, len(matched_goals)):
#     start = np.array(matched_goals[goal_idx - 1])
#     end = np.array(matched_goals[goal_idx])
#     goal_mask = goal_column == (goal_idx - 1)
#     num_points = np.sum(goal_mask)
#     if num_points == 0:
#         print(f"No control points for goal {goal_idx - 1}, skipping deviation calculation.")
#         continue
#     interpolated_segment = np.linspace(start, end, num_points)
#     interpolated_ideal_trajectory.extend(interpolated_segment)
#     actual_points = control_joints[goal_mask]
#     for actual_point, interpolated_point in zip(actual_points, interpolated_segment):
#         distance = np.linalg.norm(actual_point - interpolated_point)
#         deviations.append(distance)


# import numpy as np
# import matplotlib.pyplot as plt

# # Using actual `goal_configurations` variable
# for idx, goals in enumerate(goal_configurations):
#     # Convert to numpy array for easy processing
#     goals = np.array(goals)

#     # Calculate norms for each goal configuration
#     ideal_norms = np.linalg.norm(goals, axis=1)

#     # Create x positions for each goal
#     x_positions = range(len(ideal_norms))

#     # Plot the ideal trajectory norms
#     plt.figure(figsize=(10, 6))
#     plt.plot(x_positions, ideal_norms, linestyle='-', linewidth=3, color='#5D3A9B', label=f'Ideal Path {idx+1}')

#     # Highlight each goal
#     for i, (x, norm) in enumerate(zip(x_positions, ideal_norms)):
#         if i == 0:  # Start configuration
#             plt.scatter(x, norm, color='#34C742', marker='o', s=150, label='Start Configuration' if idx == 0 else "")
#         elif i == len(ideal_norms) - 1:  # Final configuration
#             plt.scatter(x, norm, color='#CB48EB', marker='o', s=150, label='Final Goal Configuration' if idx == 0 else "")
#         else:  # Intermediate configurations
#             plt.scatter(x, norm, color='#5D3A9B', marker='o', s=150, label='Intermediate Goal Configurations' if i == 1 and idx == 0 else "")

#     # Add labels, legend, and grid
#     plt.title(f"Ideal Goal Configurations - Path {idx+1}", fontsize=16, fontweight='bold')
#     plt.xlabel("Goal Index", fontsize=14, fontweight='bold')
#     plt.ylabel("Norm of Configurations", fontsize=14, fontweight='bold')
#     plt.legend(fontsize=12, loc='upper left')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.xlim(-2, 20)
#     plt.ylim(2, 4)

    # Show the plot
    # plt.show()

import pandas as pd

# File paths
jt_file_paths = [
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/with_obs/nn_25_astar_custom_old/20/save_distances.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/with_obs/nn_25_astar_custom_old/20/save_distances.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/with_obs/20/save_distances.csv'
]

# Process each file and count the number of configurations
for file_path in jt_file_paths:
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Exclude the first configuration (start configuration)
    number_of_configs = len(df) - 1

    # Print the result
    print(f"File: {file_path}")
    print(f"Number of configurations (excluding start): {number_of_configs}\n")


# Base file paths (use `{}` as a placeholder for the experiment number)
jt_file_paths_template = [
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/with_obs/nn_25_astar_custom_old/{}/save_distances.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/with_obs/nn_25_astar_custom_old/{}/save_distances.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/with_obs/{}/save_distances.csv'
]

# Experiment numbers
exp_no = [1, 2, 8, 9, 10, 14, 17, 18, 19, 20]

# Loop through experiments and file paths
for exp in exp_no:
    print(f"Processing experiment {exp}...\n")
    for template_path in jt_file_paths_template:
        # Format the path with the experiment number
        file_path = template_path.format(exp)

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Count configurations excluding the start configuration
            number_of_configs = len(df) - 1

            # Read the Joint columns starting from the first configuration
            joint_columns = df[['Joint 1', 'Joint 2', 'Joint 3']].to_numpy()

            # Calculate distances between consecutive configurations
            distances = np.linalg.norm(np.diff(joint_columns, axis=0), axis=1)

            # Compute the average distance
            average_distance = np.mean(distances)

            # Print the results
            print(f"File: {file_path}")
            print(f"Number of configurations (excluding start): {number_of_configs}")
            print(f"Average joint distance between consecutive configurations: {average_distance:.6f}\n")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}\n")

from fractions import Fraction

# Two decimal numbers
num1 = 0.355
num2 = 14.875

# Convert the numbers to fractions
fraction_num1 = Fraction(num1).limit_denominator()
fraction_num2 = Fraction(num2).limit_denominator()

# Compute the ratio as a fraction
ratio = fraction_num1 / fraction_num2

# Print the ratio
print(f"The ratio of {num1} to {num2} in fraction form is {ratio}")