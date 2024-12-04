import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
from fastdtw import fastdtw

# File paths
file_paths = [
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/ground_truth/astar_latest_with_obs/9/joint_angles.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_with_obs/9/joint_angles.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/euclidean/a_star_latest_with_obs/9/joint_angles.csv'
]

# Goal configurations for each file
goal_configurations = [[[-0.768775, -2.12158, 1.39752],
     [-0.704838726506011, -2.27714570417044, 1.32188321497129],
     [-0.521151727730567, -2.49630124351328, 1.23510101086307],
     [-0.204962465978053, -2.65878070027868, 1.27392171588551],
     [0.111211755510166, -2.33451616753391, 1.22824753618035],
     [0.24349083081067, -2.11630052442196, 1.227391809619],
     [0.24338503751404, -1.95420238194918, 1.23585087005368],
     [0.431265395798036, -1.76753493343225, 1.25441195852475]],
    # Learned
    [[-0.768775,	-2.12158,	1.39752],
    [-0.704838726506011,	-2.27714570417044,	1.32188321497129],
    [-0.521151727730567,	-2.49630124351328,	1.23510101086307],
    [-0.521145903112363,	-2.49626899020914,	1.44001338203889],
    [-0.07281467934491,	-2.27763622362926,	1.53236517335246],
    [0.24325049313746,	-1.95413554185669,	1.47523800621174],
    [0.24325615241488,	-1.95414614711542,	1.33920070298398],
    [0.431265395798036,	-1.76753493343225,	1.25441195852475]],
    # Image Space
    [[-0.768775,	-2.12158,	1.39752],
    [-0.837280081415277,	-2.33395009835736,	1.67264287808122],
    [-0.832860045070073,	-2.41609911191853,	1.6466021080983],
    [-0.832806814878911,	-2.57853990565455,	1.81159128257695],
    [-0.832790027285222,	-2.74053045915301,	1.73692871353639],
    [-0.521068676028129,	-2.65843559559175,	1.40104769569737],
    [-0.204968316445493,	-2.4317392553528,	0.955237274607876],
    [-0.072904536132976,	-2.23967300196191,	0.654320447670373],
    [0.111233799460399,	-2.07391148853049,	0.95558492064946],
    [0.243330149811655,	-1.93008260697712,	0.65551411875138],
    [0.43135150787286,	-1.72442960764507,	0.322237522319091],
    [0.427017182606599,	-1.74294208188325,	0.946997097717391],
    [0.431265395798036,	-1.76753493343225,	1.25441195852475]]]


# Load the control data
control_data = [pd.read_csv(fp, header=758) for fp in file_paths]

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
plt.legend(
    # fontsize=28,
    loc='upper right',
    frameon=True,
    fancybox=True,
    shadow=True,
    # title_fontsize=28,
    edgecolor='black',
    labelspacing=0.8,
    prop={'size':14, 'weight': 'bold'},  # Make the legend text bold
)
# plt.grid()
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
    plt.xlim(-100, 3500)
    plt.ylim(1.5, 3.5)
    plt.legend(
    fontsize=28,
    loc='upper right',
    frameon=True,
    fancybox=True,
    shadow=True,
    title_fontsize=28,
    edgecolor='black',
    labelspacing=1.2,
    prop={'size':18, 'weight': 'bold'},  # Make the legend text bold
)
    # plt.grid()
    plt.show()

