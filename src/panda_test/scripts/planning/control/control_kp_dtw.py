import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw  # Fast implementation of DTW
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

save_directory = "/media/jc-merlab/Crucial X9/paper_data/trajectory_pics/"
# File paths
file_paths = [
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/ground_truth/astar_latest_no_obs/3/cp.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_no_obs/3/cp.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/euclidean/astar_latest_no_obs/3/cp.csv'
]

# Load the CSV file
# Goal configurations for each file
goal_configurations = [
    # Ground Truth
    [
        [[169.0, 235.0], [188.0, 212.0], [244.0, 81.0], [242.0, 50.0], [280.0, 47.0]],
[[178.0, 221.0], [201.0, 201.0], [277.0, 76.0], [276.0, 44.0], [316.0, 43.0]],
[[178.0, 221.0], [201.0, 201.0], [278.0, 76.0], [277.0, 44.0], [317.0, 43.0]],
[[209.0, 202.0], [238.0, 190.0], [364.0, 113.0], [379.0, 83.0], [416.0, 101.0]],
[[230.0, 195.0], [260.0, 189.0], [407.0, 167.0], [430.0, 143.0], [460.0, 172.0]],
[[245.0, 193.0], [276.0, 190.0], [422.0, 220.0], [455.0, 208.0], [470.0, 248.0]],
        [[245.0, 193.0], [276.0, 190.0], [422.0, 220.0], [455.0, 208.0], [470.0, 248.0]]
],
    # Learned
    [
        [[169.0, 235.0], [188.0, 212.0], [244.0, 81.0], [242.0, 50.0], [280.0, 47.0]],
[[178.0, 221.0], [201.0, 201.0], [302.0, 94.0], [308.0, 62.0], [348.0, 69.0]],
[[209.0, 202.0], [238.0, 190.0], [385.0, 179.0], [414.0, 160.0], [437.0, 196.0]],
[[230.0, 195.0], [260.0, 189.0], [409.0, 190.0], [439.0, 176.0], [457.0, 215.0]],
    [[245.0, 193.0], [276.0, 190.0], [422.0, 220.0], [455.0, 208.0], [470.0, 248.0]]

    ],
    # Image Space
    [
        [[169.0, 235.0], [188.0, 212.0], [244.0, 81.0], [242.0, 50.0], [280.0, 47.0]],
[[167.0, 232.0], [188.0, 209.0], [259.0, 82.0], [267.0, 52.0], [306.0, 62.0]],
[[167.0, 231.0], [188.0, 209.0], [265.0, 85.0], [276.0, 56.0], [313.0, 70.0]],
[[166.0, 232.0], [187.0, 209.0], [275.0, 93.0], [295.0, 67.0], [327.0, 92.0]],
[[167.0, 231.0], [188.0, 209.0], [285.0, 100.0], [311.0, 80.0], [336.0, 113.0]],
[[166.0, 232.0], [187.0, 209.0], [293.0, 107.0], [321.0, 91.0], [341.0, 127.0]],
[[167.0, 231.0], [188.0, 209.0], [301.0, 116.0], [330.0, 100.0], [350.0, 137.0]],
[[166.0, 232.0], [187.0, 209.0], [307.0, 125.0], [336.0, 109.0], [356.0, 145.0]],
[[167.0, 231.0], [188.0, 209.0], [314.0, 134.0], [343.0, 118.0], [364.0, 154.0]],
[[167.0, 231.0], [188.0, 209.0], [324.0, 153.0], [351.0, 134.0], [375.0, 168.0]],
[[167.0, 231.0], [188.0, 209.0], [325.0, 157.0], [353.0, 138.0], [376.0, 172.0]],
[[178.0, 221.0], [201.0, 201.0], [344.0, 168.0], [375.0, 152.0], [394.0, 190.0]],
[[195.0, 208.0], [222.0, 193.0], [369.0, 196.0], [392.0, 172.0], [422.0, 201.0]],
[[209.0, 202.0], [238.0, 190.0], [385.0, 203.0], [419.0, 204.0], [418.0, 247.0]],
[[230.0, 195.0], [260.0, 189.0], [406.0, 214.0], [439.0, 225.0], [425.0, 265.0]],
    [[245.0, 193.0], [276.0, 190.0], [422.0, 220.0], [455.0, 208.0], [470.0, 248.0]]

    ]
]

# Load control data from each file
control_data = [pd.read_csv(fp, header=48) for fp in file_paths]

# Helper functions
def calculate_norm(configurations):
    """Calculates the Euclidean norm of the 5 keypoints for each configuration."""
    return np.linalg.norm(configurations, axis=(1, 2))

def interpolate_norms_continuous(start_config, goal_config, num_points):
    """Interpolates the norms of the start and goal configurations as a continuous trajectory."""
    start_norm = np.linalg.norm(start_config)
    goal_norm = np.linalg.norm(goal_config)
    return np.linspace(start_norm, goal_norm, num_points)

# Colors and labels for plotting
colors = ['#40B0A6', '#5D3A9B', '#D41159']
labels = ['Ground Truth', 'Learned', 'Image Space']

# Plot setup for combined figure
plt.figure(figsize=(20, 15))

# Process each type of data
for idx, (data, goals, color, method) in enumerate(zip(control_data, goal_configurations, colors, labels)):
    print("Method", method)
    # Extract goal column and keypoints from control data
    goal_column = data.iloc[:, 0].values
    control_keypoints = data.iloc[:, 1:11].to_numpy().reshape(-1, 5, 2)

    # Calculate actual norms
    actual_norms = calculate_norm(control_keypoints)

    # Calculate ideal norms
    ideal_norms = []
    start_config = np.array(goals[0])
    for goal_idx in range(1, len(goals)):
        start = np.array(goals[goal_idx - 1])
        goal = np.array(goals[goal_idx])
        num_points = len(control_keypoints[goal_column == (goal_idx - 1)])
        interpolated_norms = interpolate_norms_continuous(start, goal, num_points)
        ideal_norms.extend(interpolated_norms)
    ideal_norms = np.array(ideal_norms)

    # Ensure both paths start from the same configuration
    start_norm = np.linalg.norm(start_config)
    actual_norms[0] = start_norm
    ideal_norms[0] = start_norm

    # Extract goal points
    ideal_goal_points = [start_config] + goals[1:]
    actual_goal_points = [start_config]
    for goal_idx in range(1, len(goals)):
        print("Current Goal", goal_idx)
        last_row = control_keypoints[goal_column == (goal_idx - 1)][-1]
        # last_row = control_keypoints[goal_mask][-1]
        actual_goal_points.append(last_row)
    ideal_goal_points = np.array(ideal_goal_points)
    actual_goal_points = np.array(actual_goal_points)

    # Plot actual and ideal norms
    plt.plot(
        range(len(actual_norms)),
        actual_norms,
        linestyle='--',
        linewidth=5,
        color=color,
        alpha=0.9,
        label=f"{method} - Control Trajectory"
    )
    plt.plot(
        range(len(ideal_norms)),
        ideal_norms,
        linestyle='-',
        linewidth=5,
        color=color,
        label=f"{method} - Planned Path"
    )

    # Highlight start and goal points
    for g_idx, (ideal_point, actual_point) in enumerate(zip(ideal_goal_points, actual_goal_points)):
        ideal_norm = np.linalg.norm(ideal_point)
        actual_norm = np.linalg.norm(actual_point)
        x_position = sum(len(control_keypoints[goal_column == (i - 1)]) for i in range(1, g_idx + 1)) - 1

        # Plot start, intermediate, and goal points
        if g_idx == 0:  # Start configuration
            plt.scatter(x_position, ideal_norm, color='#34C742', marker='o', s=150, label="Start Configuration" if idx == 0 else '')
            plt.scatter(x_position, actual_norm, color='#34C742', marker='o', s=150)
        elif g_idx == len(ideal_goal_points) - 1:  # Final goal configuration
            plt.scatter(x_position, ideal_norm, color='#CB48EB', marker='o', s=150, label="Final Goal Configuration" if idx == 0 else '')
            plt.scatter(x_position, actual_norm, color='#CB48EB', marker='o', s=150)
        else:  # Intermediate configurations
            plt.scatter(x_position, ideal_norm, color=color, marker='o', s=150, label="")
            plt.scatter(x_position, actual_norm, color=color, marker='o', s=150, label="Intermediate Goal Configurations in Path" if idx == 0 else "")

        # Draw dashed lines between actual and ideal points
        plt.plot([x_position, x_position], [ideal_norm, actual_norm], linestyle='--', color='gray')
        distance = np.abs(ideal_norm - actual_norm)
        plt.text(x_position + 0.5, (ideal_norm + actual_norm) / 2, f"{distance:.2f}", fontsize=14, ha='left')

# Add legend, labels, and title for combined plot
plt.title("Norm-Based Continuous Comparison for Keypoints")
plt.xlabel("Total Number of Control Points")
plt.ylabel("Norm of Keypoints")
# plt.legend(loc='lower right', fontsize=14)
save_path = os.path.join(save_directory, "kp_traj_comp_obs_09a.svg")
# plt.savefig(save_path)
plt.show()

#Separate figure for each type
for idx, (data, goals, color, label) in enumerate(zip(control_data, goal_configurations, colors, labels)):
    plt.figure(figsize=(10, 10))
    
    # Extract goal column and keypoints
    goal_column = data.iloc[:, 0].values
    control_keypoints = data.iloc[:, 1:11].to_numpy().reshape(-1, 5, 2)
    
    # Align goal_column with keypoints
    goal_column = goal_column[:control_keypoints.shape[0]]
    
    # Calculate norms
    actual_norms = calculate_norm(control_keypoints)
    ideal_norms = []
    
    # Extract goal points for highlighting
    ideal_goal_points = [np.array(goals[0])] + [np.array(goal) for goal in goals[1:]]
    actual_goal_points = [np.array(goals[0])]
    
    for goal_idx in range(1, len(goals)):
        start = np.array(goals[goal_idx - 1])
        goal = np.array(goals[goal_idx])
        
        # Boolean mask for rows belonging to this goal
        goal_mask = goal_column == (goal_idx - 1)
        num_points = np.sum(goal_mask)
        
        if num_points == 0:
            raise ValueError(f"No rows found for goal {goal_idx - 1}.")
        
        interpolated_norms = interpolate_norms_continuous(start, goal, num_points)
        ideal_norms.extend(interpolated_norms)
        
        # Extract last actual configuration for the goal
        last_row = control_keypoints[goal_mask][-1]
        actual_goal_points.append(last_row)
    
    # Convert to numpy array
    ideal_norms = np.array(ideal_norms)
    start_norm = np.linalg.norm(np.array(goals[0]))
    actual_norms[0] = start_norm
    ideal_norms[0] = start_norm
    
    # Plot actual and ideal norms
    plt.plot(
        range(len(actual_norms)), actual_norms,
        linestyle='--', linewidth=5, color=color,
        alpha=0.7, label=f"{label} - Control Trajectory"
    )
    plt.plot(
        range(len(ideal_norms)), ideal_norms,
        linestyle='-', linewidth=5, color=color,
        label=f"{label} - Planned Path"
    )
    
    # Highlight goal points
    for g_idx, (ideal_point, actual_point) in enumerate(zip(ideal_goal_points, actual_goal_points)):
        ideal_norm = np.linalg.norm(ideal_point)
        actual_norm = np.linalg.norm(actual_point)

        # print("Ideal Norm", ideal_norm)
        # print("Actual norm", actual_norm)
        # print("Ideal point", ideal_point)
        # print("Actual point", actual_point)
        
        x_position = sum(len(control_keypoints[goal_column == (i - 1)]) for i in range(1, g_idx + 1)) - 1
        
        # Mark start, intermediate, and final goals
        if g_idx == 0:  # Start configuration
            plt.scatter(x_position, ideal_norm, color='#34C742', marker='o', s=250, label="Start Configuration")
            plt.scatter(x_position, actual_norm, color='#34C742', marker='o', s=250)
        elif g_idx == len(ideal_goal_points) - 1:  # Final goal configuration
            plt.scatter(x_position, ideal_norm, color='#CB48EB', marker='o', s=250, label="Final Goal Configuration")
            plt.scatter(x_position, actual_norm, color='#CB48EB', marker='o', s=250)
        else:  # Intermediate goals
            # plt.scatter(x_position, ideal_norm, color=color, marker='o', s=150)
            # plt.scatter(x_position, actual_norm, color=color, marker='o', s=150)
            plt.scatter(x_position, ideal_norm, color=color, marker='o', s=250, label="")
            plt.scatter(x_position, actual_norm, color=color, marker='o', s=250, label="Intermediate Goal Configurations in Path" if g_idx == 1 else "")
        
        # Draw dashed lines between ideal and actual points
        plt.plot([x_position, x_position], [ideal_norm, actual_norm], linestyle='--', color='gray')
        distance = np.abs(ideal_norm - actual_norm)
        plt.text(x_position + 0.5, (ideal_norm + actual_norm) / 2, f"{distance:.2f}", fontsize=14, ha='left')
    
    # Add title, labels, legend
    # plt.title(f"Norm-Based Continuous Comparison: {label}")
    # plt.xlabel("Control Points")
    # plt.ylabel("Norm of Keypoints")
    plt.xlim(-100, 2300)
    plt.ylim(550, 1200)
    plt.legend(
    fontsize=24,
    loc='lower right',
    frameon=True,
    fancybox=True,
    shadow=True,
    title_fontsize=24,
    edgecolor='black',
    labelspacing=1.2,
    prop={'size':18, 'weight': 'bold'},  # Make the legend text bold
)
    # plt.grid()
    # plt.tick_params(axis='both', which='major', labelsize=18)
    for label in plt.gca().get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(28)
    for label in plt.gca().get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(28)

    if isinstance(label, str):
        label_text = label
    else:
        label_text = label.get_text() if hasattr(label, 'get_text') else str(label)
    # plt.legend(fontsize=14, loc='upper left')
    # Save the figure
    save_path = os.path.join(save_directory, f"kp_{label.get_text().replace(' ', '_').lower()}_traj_obs_03_paper.svg")
    # plt.savefig(save_path, bbox_inches='tight')
    print(f"Figure saved as {save_path}")
    plt.show()

# Initialize variables to store statistics
deviation_stats = []

# Loop through each roadmap
for idx, (data, goals, color, label) in enumerate(zip(control_data, goal_configurations, colors, labels)):
    plt.figure(figsize=(20, 15))

    # Extract goal column and keypoints
    goal_column = data.iloc[:, 0].values
    control_keypoints = data.iloc[:, 1:11].to_numpy().reshape(-1, 5, 2)

    print(control_keypoints.shape[0])

    # Align goal_column with keypoints
    goal_column = goal_column[:control_keypoints.shape[0]]

    # Interpolate ideal trajectories
    interpolated_ideal_trajectory = []
    deviations = []
    for goal_idx in range(1, len(goals)):
        start = np.array(goals[goal_idx - 1])
        end = np.array(goals[goal_idx])
        # Boolean mask for rows belonging to this goal
        goal_mask = goal_column == (goal_idx - 1)
        num_points = np.sum(goal_mask)
        
        if num_points == 0:
            raise ValueError(f"No rows found for goal {goal_idx - 1}.")
        
        # Interpolate keypoints
        interpolated_segment = np.linspace(start, end, num_points)
        interpolated_ideal_trajectory.extend(interpolated_segment)

        # Calculate deviations for each point
        actual_points = control_keypoints[goal_mask]
        for actual_point, interpolated_point in zip(actual_points, interpolated_segment): 
            # print(actual_point)
            # print(interpolated_point)
            distance = np.linalg.norm(actual_point - interpolated_point)
            print(distance)
            deviations.append(distance)

    interpolated_ideal_trajectory = np.array(interpolated_ideal_trajectory)

    # Compute statistics
    total_deviation = sum(deviations)
    max_deviation = max(deviations)
    total_points = len(deviations)
    deviation_stats.append({
        "Roadmap": label,
        "Total Points": total_points,
        "Total Deviation": total_deviation,
        "Max Deviation": max_deviation
    })

    
deviation_stats_df = pd.DataFrame(deviation_stats)
print(deviation_stats_df)