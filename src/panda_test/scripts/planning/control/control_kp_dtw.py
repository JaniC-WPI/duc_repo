import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw  # Fast implementation of DTW
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load the CSV file
csv_file_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_with_obs/20/cp.csv'
control_data = pd.read_csv(csv_file_path, header=0)  # Adjust 'header' if necessary

def calculate_norm(configurations):
    """
    Calculates the Euclidean norm of the 5 keypoints for each configuration.
    Args:
        configurations: A numpy array of shape (N, 5, 2), where N is the number of configurations.
    Returns:
        A numpy array of shape (N,) representing the norm for each configuration.
    """

    norm_1 = np.sqrt(np.sum(configurations**2, axis=(1, 2)))
    norm_2 = np.linalg.norm(configurations, axis=(1, 2))
    return norm_2

# def interpolate_configurations(start, goal, num_points):
#     """
#     Dynamically interpolates between two configurations for all keypoints to match the number of points.
#     Args:
#         start: Starting configuration (5 keypoints with x, y coordinates).
#         goal: Goal configuration (5 keypoints with x, y coordinates).
#         num_points: Number of points to interpolate (matches control data rows).
#     Returns:
#         Interpolated configurations as a numpy array of shape (num_points, 5, 2).
#     """
#     start = np.array(start)
#     goal = np.array(goal)
#     interpolated_path = []
    
#     for keypoint_idx in range(start.shape[0]):  # Loop over each keypoint
#         x_interp = interp1d([0, num_points - 1], [start[keypoint_idx, 0], goal[keypoint_idx, 0]])
#         y_interp = interp1d([0, num_points - 1], [start[keypoint_idx, 1], goal[keypoint_idx, 1]])
#         keypoint_path = [(x_interp(t), y_interp(t)) for t in range(num_points)]
#         interpolated_path.append(keypoint_path)

        # return np.array(interpolated_path).transpose(1, 0, 2)

def interpolate_norms_continuous(start_config, goal_config, num_points):
    """
    Interpolates the norms of the start and goal configurations as a continuous trajectory.
    Args:
        start_config: Starting configuration (5 keypoints with x, y coordinates).
        goal_config: Goal configuration (5 keypoints with x, y coordinates).
        num_points: Number of points to interpolate (matches control data rows).
    Returns:
        A numpy array of interpolated norms with shape (num_points,).
    """
    start_norm = np.linalg.norm(start_config)
    goal_norm = np.linalg.norm(goal_config)
    return np.linspace(start_norm, goal_norm, num_points)

        

# Extract goal column and reshape keypoints into (5, 2) structure per row
goal_column = control_data.iloc[:, 0].values
control_keypoints = control_data.iloc[:, 1:11].to_numpy().reshape(-1, 5, 2)

# Define the goal configurations (start, intermediate, and final)
goal_configurations = [
    [[314.0, 210.0], [339.0, 224.0], [452.0, 129.0], [474.0, 105.0], [505.0, 132.0]],
    [[303.0, 203.0], [331.0, 215.0], [426.0, 101.0], [444.0, 73.0], [479.0, 96.0]],
    [[303.0, 203.0], [331.0, 215.0], [426.0, 101.0], [449.0, 76.0], [479.0, 104.0]],
    [[267.0, 193.0], [298.0, 196.0], [373.0, 71.0], [395.0, 46.0], [426.0, 72.0]],
    [[231.0, 195.0], [261.0, 189.0], [329.0, 60.0], [349.0, 35.0], [381.0, 61.0]],
    [[209.0, 202.0], [238.0, 190.0], [313.0, 64.0], [333.0, 38.0], [366.0, 63.0]],
    [[196.0, 208.0], [223.0, 193.0], [300.0, 70.0], [319.0, 44.0], [352.0, 68.0]],
    [[167.0, 232.0], [188.0, 209.0], [246.0, 77.0], [260.0, 48.0], [296.0, 65.0]],
    [[147.0, 262.0], [160.0, 234.0], [197.0, 96.0], [204.0, 65.0], [243.0, 73.0]],
    [[138.0, 283.0], [146.0, 253.0], [182.0, 112.0], [189.0, 80.0], [228.0, 88.0]],
    [[141.0, 291.0], [147.0, 261.0], [179.0, 123.0], [185.0, 92.0], [225.0, 99.0]]
]

# # Results dictionary
# results = {}

# # Compare paths for each goal
# for goal_idx in range(1, len(goal_configurations)):  # Skip the start configuration
#     # Define start and goal configurations
#     start = goal_configurations[goal_idx - 1]
#     goal = goal_configurations[goal_idx]

#     # Extract the actual path for this goal from control data
#     actual_path = control_keypoints[goal_column == (goal_idx - 1)]

#     # Dynamically interpolate the ideal path to match the number of actual points
#     interpolated_path = interpolate_configurations(start, goal, len(actual_path))

#     # Compare each interpolated point with the corresponding actual point
#     point_differences = np.linalg.norm(interpolated_path - actual_path, axis=2)  # Euclidean distance

#     # Compare the last row of actual path with the goal configuration
#     final_actual = actual_path[-1]
#     final_goal = np.array(goal)
#     final_differences = np.linalg.norm(final_actual - final_goal, axis=1)  # Per keypoint

#     # Store results
#     results[f"Goal {goal_idx}"] = {
#         "point_differences": point_differences,  # Differences for each point
#         "final_differences": final_differences,  # Final row comparison
#     }
# Compute norms for the entire control data (actual trajectory)
# Compute norms for the entire control data (actual trajectory)
actual_norms = calculate_norm(control_keypoints)

# Compute ideal norms for the entire trajectory (goal-based interpolation)
ideal_norms = []
start_config = np.array(goal_configurations[0])

for goal_idx in range(1, len(goal_configurations)):
    # Define start and goal configurations for the current goal
    start = np.array(goal_configurations[goal_idx - 1])
    goal = np.array(goal_configurations[goal_idx])
    
    # Extract the number of points for this goal from the control data
    num_points = len(control_keypoints[goal_column == (goal_idx - 1)])
    
    # Interpolate norms for this goal
    interpolated_norms = interpolate_norms_continuous(start, goal, num_points)
    ideal_norms.extend(interpolated_norms)  # Append to the ideal norms list

# Convert ideal norms to a numpy array for consistency
ideal_norms = np.array(ideal_norms)

# Add the same starting norm for both trajectories
start_norm = np.linalg.norm(start_config)
actual_norms[0] = start_norm  # Ensure the actual trajectory starts from the same point
ideal_norms[0] = start_norm


# Extract the start and goal points for both paths
ideal_goal_points = [start_config] + goal_configurations[1:]  # All configurations in `goal_configurations`
actual_goal_points = [start_config]  # Start with the shared start configuration

for goal_idx in range(1, len(goal_configurations)):
    # Extract the last row for the current goal from control data
    last_row = control_keypoints[goal_column == (goal_idx - 1)][-1]
    actual_goal_points.append(last_row)

# Convert to numpy arrays for consistency
ideal_goal_points = np.array(ideal_goal_points)
actual_goal_points = np.array(actual_goal_points)

plt.figure(figsize=(12, 8))

# Plot the actual norms
plt.plot(
    range(len(actual_norms)),
    actual_norms,
    linestyle='--',
    linewidth=2,
    label="Actual Norm Trajectory"
)

# Plot the ideal norms
plt.plot(
    range(len(ideal_norms)),
    ideal_norms,
    linestyle='-',
    linewidth=2,
    label="Ideal Norm Trajectory"
)

# Highlight the start and goal points for both paths and add dashed lines for distances
for idx, (ideal_point, actual_point) in enumerate(zip(ideal_goal_points, actual_goal_points)):
    # Calculate the norm of the goal points
    ideal_norm = np.linalg.norm(ideal_point)
    actual_norm = np.linalg.norm(actual_point)
    
    # Compute the x-position for the goal points
    x_position = sum(len(control_keypoints[goal_column == (i - 1)]) for i in range(1, idx + 1)) - 1
    
    # Plot the ideal point
    plt.scatter(
        x_position,
        ideal_norm,
        color='red', marker='o', s=150,
        label="Ideal Start/Goal" if idx == 0 else ""
    )
    
    # Plot the actual point
    plt.scatter(
        x_position,
        actual_norm,
        color='blue', marker='o', s=150,
        label="Actual Start/Goal" if idx == 0 else ""
    )
    
    # Draw a dashed line between corresponding goal points
    plt.plot(
        [x_position, x_position],
        [ideal_norm, actual_norm],
        linestyle='--',
        color='gray'
    )
    
    # Calculate and annotate the distance between the actual and ideal goals
    distance = np.abs(ideal_norm - actual_norm)
    plt.text(
        x_position + 0.5,  # Offset to avoid overlap
        (ideal_norm + actual_norm) / 2,  # Midpoint of the line
        f"{distance:.2f}",
        color='black',
        fontsize=10,
        ha='left'
    )

# Add title, labels, and legend
plt.title("Robot Trajectory: Norm-Based Continuous Comparison with Goal Distances")
plt.xlabel("Total Number of Control Points")
plt.ylabel("Norm of Keypoints")
plt.legend()
# plt.grid()
plt.show()


# # Results dictionary
# results = {}

# # Create a single plot for the norm trajectories
# plt.figure(figsize=(12, 8))

# # Plot the actual norms
# plt.plot(
#     range(len(actual_norms)),
#     actual_norms,
#     linestyle='--',
#     linewidth=2,
#     label="Actual Norm Trajectory"
# )

# # Plot the ideal norms
# plt.plot(
#     range(len(ideal_norms)),
#     ideal_norms,
#     linestyle='-',
#     linewidth=2,
#     label="Ideal Norm Trajectory"
# )

# # Highlight the start and goal points for both paths
# for idx, (ideal_point, actual_point) in enumerate(zip(ideal_goal_points, actual_goal_points)):
#     # Calculate the norm of the goal points
#     ideal_norm = np.linalg.norm(ideal_point)
#     actual_norm = np.linalg.norm(actual_point)
    
#     # Compute the x-position for the goal points
#     x_position = sum(len(control_keypoints[goal_column == (i - 1)]) for i in range(1, idx + 1)) - 1
    
#     # Plot the ideal point
#     plt.scatter(
#         x_position,
#         ideal_norm,
#         color='red', marker='o', s=150,
#         label="Ideal Start/Goal" if idx == 0 else ""
#     )
    
#     # Plot the actual point
#     plt.scatter(
#         x_position,
#         actual_norm,
#         color='blue', marker='o', s=150,
#         label="Actual Start/Goal" if idx == 0 else ""
#     )

# # Add title, labels, and legend
# plt.title("Robot Trajectory: Norm-Based Continuous Comparison (Ideal vs Actual)")
# plt.xlabel("Total Number of Control Points")
# plt.ylabel("Norm of Keypoints")
# plt.legend()
# plt.grid()
# plt.show()


# # Visualize ideal vs. actual paths for each goal
# # Create a single plot for all goals
# # Create a single plot for all goals
# plt.figure(figsize=(12, 8))

# # Loop through all goals
# for goal_idx in range(1, len(goal_configurations)):  # Skip the start configuration
#     # Define start and goal configurations
#     start = goal_configurations[goal_idx - 1]
#     goal = goal_configurations[goal_idx]
    
#     # Extract the actual path for this goal
#     actual_path = control_keypoints[goal_column == (goal_idx - 1)]
    
#     # Dynamically interpolate the ideal path to match the number of actual points
#     interpolated_path = interpolate_configurations(start, goal, len(actual_path))
    
#     # Plot trajectories for each keypoint
#     for keypoint_idx in range(5):  # Loop through each keypoint
#         # Plot actual path for the keypoint
#         # Highlight the start and goal points for the actual path
#         plt.scatter(
#             actual_path[0, keypoint_idx, 0], 
#             actual_path[0, keypoint_idx, 1], 
#             color='red', marker='x', s = 200, label = ""
#         )
#         plt.scatter(
#             actual_path[-1, keypoint_idx, 0], 
#             actual_path[-1, keypoint_idx, 1], 
#             color='blue', marker='x', s = 200, label = ""
#         )
        
#         # Highlight the start and goal points for the ideal path
#         plt.scatter(
#             interpolated_path[0, keypoint_idx, 0], 
#             interpolated_path[0, keypoint_idx, 1], 
#             color='red', marker='o', s = 100, label="" 
#         )
#         plt.scatter(
#             interpolated_path[-1, keypoint_idx, 0], 
#             interpolated_path[-1, keypoint_idx, 1], 
#             color='blue', marker='o', s = 100, label = ""
#         )
#         plt.plot(
#             actual_path[:, keypoint_idx, 0], 
#             actual_path[:, keypoint_idx, 1], 
#             linestyle='-', linewidth=4, label=f"Keypoint {keypoint_idx + 1} Actual" if goal_idx == 1 else ""
#         )
        
#         # Plot ideal interpolated path for the keypoint
#         plt.plot(
#             interpolated_path[:, keypoint_idx, 0], 
#             interpolated_path[:, keypoint_idx, 1], 
#             linestyle='-', linewidth=4, label=f"Keypoint {keypoint_idx + 1} Ideal" if goal_idx == 1 else ""
#         )
        
        

# # Add title, labels, and legend
# plt.title("Robot Trajectory: Ideal vs Actual Paths with Highlighted Starts and Goals")
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.gca().invert_yaxis()
# plt.legend()
# plt.grid()
# plt.show()