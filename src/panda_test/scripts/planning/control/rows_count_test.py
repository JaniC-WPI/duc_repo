import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Save directory
save_directory = "/media/jc-merlab/Crucial X9/paper_data/trajectory_pics/"

# File paths
file_paths = [
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/ground_truth/astar_latest_with_obs/18/joint_angles.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_with_obs/18/joint_angles.csv'
    # '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/euclidean/astar_latest_with_obs/18/joint_angles.csv'
]

kp_file_paths = [
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/ground_truth/astar_latest_with_obs/18/cp.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_with_obs/18/cp.csv'
    # '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/euclidean/astar_latest_with_obs/18/cp.csv'
]

jt_file_paths = [
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/with_obs/nn_25_astar_custom_old/18/save_distances.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/with_obs/nn_25_astar_custom_old/18/save_distances.csv'
    # '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/with_obs/18/save_distances.csv'
]

# Function Definitions
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

initial_skip = 50

def get_first_goal_row_count(kp_file, initial_skip):
    """Get the row count for the first goal after skipping the header and initial rows."""
    kp_df = pd.read_csv(kp_file, skiprows=initial_skip, header=0)
    goal_columns = kp_df.iloc[:, 0].to_numpy()
    first_goal = goal_columns[0]
    kp_rows = (goal_columns == first_goal).sum()
    print(f"First goal rows in {kp_file}: {kp_rows} (after skipping {initial_skip} rows).")
    return kp_rows

def get_last_goal_row_count(kp_file, initial_skip):
    """Get the row count for the last goal after skipping rows."""
    kp_df = pd.read_csv(kp_file, skiprows=initial_skip, header=0)
    goal_columns = kp_df.iloc[:, 0].to_numpy()
    unique_goals = np.unique(goal_columns)
    last_goal = unique_goals[-1]
    last_goal_rows = (goal_columns == last_goal).sum()
    print(f"Last goal rows in {kp_file}: {last_goal_rows} (after skipping {initial_skip} rows).")
    return last_goal_rows

def get_total_goal_rows(control_file):
    """Get the total rows for the first and last goals in control_file."""
    control_df = pd.read_csv(control_file, header=None)
    goal_columns = control_df.iloc[:, 0].to_numpy()
    first_goal = goal_columns[0]
    last_goal = goal_columns[-1]
    total_first_goal = (goal_columns == first_goal).sum()
    total_last_goal = (goal_columns == last_goal).sum()
    return total_first_goal, total_last_goal

def find_closest_to_ideal_last_goal(control_joints, ideal_last_goal_norm):
    """Find the index in control_joints with the closest norm to the ideal last goal norm."""
    norms = np.linalg.norm(control_joints, axis=1)
    closest_index = np.argmin(np.abs(norms - ideal_last_goal_norm))
    closest_value = control_joints[closest_index]  # Joint values at the closest index

    print(f"Closest index to ideal last goal norm ({ideal_last_goal_norm:.3f}): {closest_index}")
    print(f"Closest last goal value: {closest_value}")
    return closest_index

def read_control_data_limited(file_path, kp_first_rows, kp_last_rows, total_first_goal, total_last_goal, ideal_last_goal):
    """Read control data with adjusted rows for first goal and closest last goal."""
    # Adjust first goal
    skip_first = total_first_goal - (kp_first_rows * 3)
    print(f"Skipping {skip_first} rows for first goal in {file_path}")

    control_df = pd.read_csv(file_path, skiprows=skip_first, header=None)

    # Limit rows for the last goal
    goal_columns = control_df.iloc[:, 0].to_numpy()
    last_goal = goal_columns[-1]
    last_goal_indices = np.where(goal_columns == last_goal)[0]
    limit_last_rows = kp_last_rows * 3
    last_goal_indices_limited = last_goal_indices[:limit_last_rows]

    print(f"Limiting last goal rows to {limit_last_rows} in {file_path}")

    # Find the closest index to the ideal last goal norm
    last_goal_joints = control_df.iloc[last_goal_indices_limited, 1:4].to_numpy()
    ideal_last_goal_norm = np.linalg.norm(ideal_last_goal)
    closest_index = find_closest_to_ideal_last_goal(last_goal_joints, ideal_last_goal_norm)

    # Combine earlier rows and the closest last goal row
    rows_to_keep = np.concatenate([np.where(goal_columns != last_goal)[0], last_goal_indices_limited[:closest_index + 1]])
    control_df_limited = control_df.iloc[rows_to_keep]

    print(f"Final control data rows: {len(control_df_limited)}")
    return control_df_limited

# Process Each File Pair
control_data = []

for idx, (kp_file, file_path) in enumerate(zip(kp_file_paths, file_paths)):
    print(f"\nProcessing: {kp_file} and {file_path}")
    
    # Step 1: Get first and last goal row counts
    kp_first_rows = get_first_goal_row_count(kp_file, initial_skip=50)
    kp_last_rows = get_last_goal_row_count(kp_file, initial_skip=50)

    # Step 2: Get total rows for first and last goals
    total_first_goal, total_last_goal = get_total_goal_rows(file_path)

    # Step 3: Use the last goal from goal_configurations
    ideal_last_goal = goal_configurations[idx][-1]

    # Step 4: Read control data
    control_df = read_control_data_limited(
        file_path, kp_first_rows, kp_last_rows, total_first_goal, total_last_goal, ideal_last_goal
    )
    control_data.append(control_df)

print("\nFinished processing all files.")

labels = ['Ground Truth', 'Learned']
deviation_stats = []

# Loop through each roadmap
for idx, (data, goals, label) in enumerate(zip(control_data, goal_configurations, labels)):
    plt.figure(figsize=(20, 15))

    # Extract goal column and joint angles
    goal_column = data.iloc[:, 0].values
    control_joints = data.iloc[:, 1:4].to_numpy()

    print(control_joints.shape[0])

    goal_column = goal_column[:control_joints.shape[0]]

    # Interpolate ideal trajectories
    interpolated_ideal_trajectory = []
    deviations = []
    for goal_idx in range(1, len(goals)):
        start = np.array(goals[goal_idx - 1])
        end = np.array(goals[goal_idx])
        # num_points = len(control_joints[goal_column == (goal_idx - 1)])
        goal_mask = goal_column == (goal_idx - 1)
        num_points = np.sum(goal_mask)
        if num_points == 0:
            raise ValueError(f"No rows found for goal {goal_idx - 1}.")
        
        interpolated_segment = np.linspace(start, end, num_points)
        interpolated_ideal_trajectory.extend(interpolated_segment)
        # Calculate deviations for each point

        actual_points = control_joints[goal_mask]

        for actual_point, interpolated_point in zip(actual_points, interpolated_segment):

            distance = np.linalg.norm(actual_point - interpolated_point)

            deviations.append(distance)

    interpolated_ideal_trajectory = np.array(interpolated_ideal_trajectory)

    # Calculate deviations
    # deviations = []
    # for point_idx in range(len(control_joints)):
    #     actual_point = control_joints[point_idx]
    #     ideal_point = interpolated_ideal_trajectory[point_idx]
    #     distance = np.linalg.norm(actual_point - ideal_point)
    #     deviations.append(distance)

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

deviation_stats_df = pd.DataFrame(deviation_stats)
print(deviation_stats_df)