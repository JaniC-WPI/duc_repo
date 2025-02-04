import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw  # Fast implementation of DTW
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# Base file paths (use `{}` as a placeholder for the experiment number) 
jt_file_paths_template = [
    # '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/no_obs/nn_25_astar_custom_old/{}/save_distances.csv',
    # '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/no_obs/nn_25_astar_custom_old/{}/save_distances.csv',
    '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/with_obs/{}/save_distances.csv'
]

# File type labels
file_types = [str('Ground Truth'), str('Learned'), str('Image Space')]

# Colors for the file types
colors=['#40B0A6', '#5D3A9B', '#D41159']  # Assign unique colors for each type

# Output directory for plots
output_dir = '/media/jc-merlab/Crucial X9/paper_data/trajectory_plots/'
os.makedirs(output_dir, exist_ok=True)

# Experiment numbers
# exp_no = [1, 2, 3, 4, 6, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20] # exps for no obs

# exp_no = [1, 2, 8, 9, 10, 14, 17, 18, 19, 20] # exps for with obs

exp_no = [1]

# Loop through experiments and file paths
for exp in exp_no:
    print(f"Processing experiment {exp}...\n")
    exp = str(exp)
    # Create a figure with three subplots (one per file type)
    # First figure for normalized distances
    fig1, axes1 = plt.subplots(3, 1, figsize=(18, 15))
    # Second figure for normalized actual keypoints and joint values
    fig2, axes2 = plt.subplots(3, 1, figsize=(18, 15))

    fig3, axes3 = plt.subplots(3, 1, figsize=(18, 15))
    # Second figure for normalized actual keypoints and joint values
    fig4, axes4 = plt.subplots(3, 1, figsize=(18, 15))

    # fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    max_num_configs = -np.inf

    for i, template_path in enumerate(jt_file_paths_template):
        # Format the path with the experiment number
        file_path = template_path.format(exp)

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Update the maximum number of configurations
            max_num_configs = max(max_num_configs, len(df))

        except Exception as e:
            print(f"Error processing file {file_path}: {e}\n")

    # Set x_lim using the max number of configurations
    x_lim = (-1, max_num_configs)

    # for template_path in jt_file_paths_template:
    for i, template_path in enumerate(jt_file_paths_template):
        # Format the path with the experiment number
        file_path = template_path.format(exp)

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Count configurations excluding the start configuration
            number_of_configs = len(df) - 1

            # Extract x and y keypoints using column indexing
            x_keypoints = df.iloc[:, 1:18:2].to_numpy()  # Odd columns: 1, 3, ..., 17
            y_keypoints = df.iloc[:, 2:19:2].to_numpy()  # Even columns: 2, 4, ..., 18

            actual_keypoint_norms = np.sqrt(np.sum((x_keypoints - y_keypoints) ** 2, axis=1))

            abs_kp_dist = np.abs(np.diff(actual_keypoint_norms))

            # Read the Joint columns starting from the first configuration
            joint_columns = df[['Joint 1', 'Joint 2', 'Joint 3']].to_numpy()

            # print(joint_columns)

            # Compute distances for each joint
            j1_distances = np.sqrt(np.diff(joint_columns[:, 0]) ** 2)  # Joint 1
            j2_distances = np.sqrt(np.diff(joint_columns[:, 1]) ** 2)  # Joint 2
            j3_distances = np.sqrt(np.diff(joint_columns[:, 2]) ** 2)  # Joint 3

            j1_dist_norm = (j1_distances - np.min(j1_distances))/(np.max(j1_distances) - np.min(j1_distances))
            j2_dist_norm = (j2_distances - np.min(j2_distances))/(np.max(j2_distances) - np.min(j2_distances))
            j3_dist_norm = (j3_distances - np.min(j3_distances))/(np.max(j3_distances) - np.min(j3_distances))

            actual_joint_norms = np.sqrt(np.sum(joint_columns ** 2, axis=1))

            abs_jt_dist = np.abs(np.diff(actual_joint_norms))

            abs_kp_dist_norm = (abs_kp_dist - np.min(abs_kp_dist)) / (np.max(abs_kp_dist) - np.min(abs_kp_dist))

            abs_jt_dist_norm = (abs_jt_dist - np.min(abs_jt_dist)) / (np.max(abs_jt_dist) - np.min(abs_jt_dist))

            # Normalize X Keypoints
            x_min, x_max = np.min(x_keypoints), np.max(x_keypoints)
            x_keypoints_normalized = (x_keypoints - x_min) / (x_max - x_min)

            # Normalize Y Keypoints
            y_min, y_max = np.min(y_keypoints), np.max(y_keypoints)
            y_keypoints_normalized = (y_keypoints - y_min) / (y_max - y_min)

            all_keypoints = np.concatenate([x_keypoints, y_keypoints], axis=1)

            kp_diff = np.diff(all_keypoints, axis=0)
            kp_distances = np.linalg.norm(kp_diff, axis=1)

            # print(f"{file_types[i]} - {exp} no_obs mean raw keypoints distances: ", np.mean(kp_distances))
            # print(f"{file_types[i]} - {exp} no_obs variance raw keypoints distances: ", np.std(kp_distances, ddof=0))


            kp_dist_direct_norm = (kp_distances - np.min(kp_distances)) / (np.max(kp_distances) - np.min(kp_distances))

            # print(f"{file_types[i]} - {exp} no_obs mean normalized keypoints distances: ", np.mean(kp_dist_direct_norm))
            # print(f"{file_types[i]} - {exp} no_obs variance normalized keypoints distances: ", np.std(kp_dist_direct_norm))

            # Combine normalized x and y keypoints
            all_keypoints_normalized = np.concatenate([x_keypoints_normalized, y_keypoints_normalized], axis=1)

            # Print normalized keypoints for verification
            # print("Normalized X Keypoints:", x_keypoints_normalized)
            # print("Normalized Y Keypoints:", y_keypoints_normalized)
            # print("Combined Normalized Keypoints:", all_keypoints_normalized)

            # Compute differences between consecutive rows
            differences = np.diff(all_keypoints_normalized, axis=0)

            # Calculate Euclidean distances
            kp_distances_norm = np.linalg.norm(differences, axis=1)

            # jt_dist_direct_norm is to take the joint distances from raw joint values and then normalize it

            jt_diff = np.diff(joint_columns, axis = 0)
            jt_distances = np.linalg.norm(jt_diff, axis=1)

            # print(f"{file_types[i]} - {exp} no_obs mean raw joint distances: ", np.mean(jt_distances))
            # print(f"{file_types[i]} - {exp} no_obs variance raw joint distances: ", np.std(jt_distances))

            jt_dist_direct_norm = (jt_distances - np.min(jt_distances)) / (np.max(jt_distances) - np.min(jt_distances))

            # print(f"{file_types[i]} - {exp} no_obs mean normalized joint distances: ", np.mean(jt_dist_direct_norm))
            # print(f"{file_types[i]} - {exp} no_obs variance normalized joint distances: ", np.std(jt_dist_direct_norm))

            dist_diff = np.abs(kp_dist_direct_norm - jt_dist_direct_norm)

            print(f"{file_types[i]} - {exp} no_obs mean differences of distances: ", np.mean(dist_diff))
            print(f"{file_types[i]} - {exp} no_obs variance differences of distances: ", np.std(dist_diff))
            

            # Normalize the actual joint distances to a range of 0 to 1
            joint_columns_normalized = (joint_columns - np.min(joint_columns)) / (np.max(joint_columns) - np.min(joint_columns))

            # print("Joint Normalized", joint_columns_normalized)
            
            # jt_distances_norm is to normalize the raw joint values and then take the distance between the normalized joints
            # Calculate distances between consecutive configurations
            jt_distances_norm = np.linalg.norm(np.diff(joint_columns_normalized, axis=0), axis=1)

            # keypoints_norms and joint_norms are the norms of each set of normalized keypoints  and nrmalized joint values for each way points
            # Calculate norms for all_keypoints_normalized
            x_coords = all_keypoints_normalized[:, :9]
            y_coords = all_keypoints_normalized[:, 9:]
            keypoint_norms = np.sqrt(np.sum((x_coords - y_coords) ** 2, axis=1))

            # Calculate norms for joint_columns_normalized
            joint_norms = np.sqrt(np.sum(joint_columns_normalized ** 2, axis=1))

            # kp_distfrom_norm, jt_dist_from_norm is to take the absolute difference between keypoint norm, and joint norm respectively

            kp_dist_from_norm = np.abs(np.diff(keypoint_norms)) 

            jt_dist_from_norm = np.abs(np.diff(joint_norms)) 

            # Combine x and y into keypoint arrays
            # keypoints = np.stack((x_keypoints, y_keypoints), axis=-1)  # Shape: (num_configs, num_keypoints, 2)

            # Calculate Euclidean distances between consecutive configurations
            # kp_distances = np.linalg.norm(keypoints[1:] - keypoints[:-1], axis=(1, 2))

            #  # Normalize the distances to a range of 0 to 1
            # jt_distances_normalized = (jt_distances - np.min(jt_distances)) / (np.max(jt_distances) - np.min(jt_distances))
            # kp_distances_normalized = (kp_distances - np.min(kp_distances)) / (np.max(kp_distances) - np.min(kp_distances))

            # print(f"File: {file_path}")
            # print(f"joint distances between consecutive configurations: {jt_distances}\n")
            # print(f"keypoints distances between consecutive configurations: {kp_distances}\n")

            # print(f"Normalized Joint Distances for experiment {exp}: {jt_distances_normalized}")
            # print(f"Normalized Keypoint Distances for experiment {exp}: {kp_distances_normalized}\n")

            # Plot in the corresponding subplot
            ax1 = axes1[i]
            ax1.plot(abs_kp_dist_norm, label='Keypoint distances', marker='o', color=colors[i], markersize=14, linewidth=4)
            ax1.plot(abs_jt_dist_norm, label='Joint distances', marker='x', linestyle='--', color=colors[i], markersize=20, markeredgewidth=5, linewidth=4)
            ax1.set_title(f'{file_types[i]} - Absolute distances {exp} no_obs', fontsize=18, fontweight='bold')
            # if i == 2:  # Add x-axis label only to the last subplot
                # ax1.set_xlabel('Configuration Pairs', fontsize=20, fontweight='bold')
            # ax1.set_xlabel('Configuration Index', fontsize=16, fontweight='bold')
            # ax1.set_ylabel('Distances', fontsize=20, fontweight='bold')
            ax1.legend(fontsize=18, prop={'weight': 'bold'})
            ax1.grid(True)
            ax1.tick_params(axis='both', labelsize=28, width=2)
            for label in ax1.get_xticklabels() + ax1.get_yticklabels():  # Make tick labels bold
                label.set_fontweight('bold')
            ax1.set_xlim(x_lim)
            # ax1.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))

            max_configs = max(len(kp_dist_direct_norm) for _ in jt_file_paths_template)

            ax2 = axes2[i]
            ax2.plot(kp_dist_direct_norm, label=f'keypoints distances \n in image (in pixels)', marker='o', color=colors[i], markersize=14, linewidth=4)
            ax2.plot(jt_dist_direct_norm, label=f'joint distances \n (in radians)', marker='x', linestyle='--', color=colors[i], markersize=20, markeredgewidth=5, linewidth=4)
            # ax2.set_title(f'{file_types[i]} - Euclidean distances {exp} no_obs', fontsize=18, fontweight='bold')
            # Add text annotations for distances between configurations            
            # if i == 2:  # Add x-axis label only to the last subplot
            #     ax2.set_xlabel('Configuration Index', fontsize=20, fontweight='bold')
            # ax2.set_ylabel('Distances', fontsize=20, fontweight='bold')
            ax2.legend(prop={'size': 20, 'weight': 'bold'}, loc='upper right', bbox_to_anchor=(1.007, 1.025))
            ax2.grid(False)
            ax2.tick_params(axis='both', labelsize=28, width=2)
            for label in ax2.get_xticklabels() + ax2.get_yticklabels():  # Make tick labels bold
                label.set_fontweight('bold')
                # Set x-tick labels starting from 1
            # Set x-ticks to match configuration indices and labels to start at 1
            x_ticks = range(0, max_num_configs, 2)  # Tick positions starting at 0
            x_labels = range(1, max_num_configs + 1, 2)  # Labels starting from 1

            ax2.set_xlim(x_lim)  # Ensure x-axis spans all configurations (0-based index)
            ax2.set_xticks(x_ticks)               # Set ticks at the correct positions

            # Ensure x-tick labels appear only for the last subplot
            if i != 2:
                ax2.set_xticklabels([])           # Clear tick labels for other subplots
            else:
                ax2.set_xticklabels([str(label) for label in x_labels])  # Labels explicitly start from 1

            # Plot normalized actual keypoints and joint values (new plots)
            ax3 = axes3[i]
            ax3.plot(keypoint_norms, label='Normalized Keypoints', marker='o', color=colors[i], markersize=14, linewidth=4)
            ax3.plot(joint_norms, label='Normalized Joints', marker='x', linestyle='--', color=colors[i], markersize=20, markeredgewidth=5, linewidth=4)
            ax3.set_title(f'{file_types[i]} - Normalized Values', fontsize=18, fontweight='bold')
            # Add text annotations for distances between configurations            
            if i == 2:  # Add x-axis label only to the last subplot
                ax3.set_xlabel('Configuration Index', fontsize=20, fontweight='bold')
            ax3.set_ylabel('Normalized Value', fontsize=20, fontweight='bold')
            ax3.legend(fontsize=18)
            ax3.grid(True)
            ax3.tick_params(axis='both', labelsize=28, width=2)
            ax3.set_xlim(x_lim)

            ax4 = axes4[i]
            ax4.plot(kp_dist_direct_norm, label='Keypoints distances', marker='o', color=colors[i], markersize=14, linewidth=4)
            ax4.plot(j1_dist_norm, label='Joint 1 distances', marker='x', linestyle='--', color=colors[i], markersize=20, markeredgewidth=5, linewidth=4)
            ax4.plot(j2_dist_norm, label='Joint 2 distances', marker='x', linestyle='-.', color=colors[i], markersize=20, markeredgewidth=5, linewidth=4)
            ax4.plot(j3_dist_norm, label='Joint 3 distances', marker='x', linestyle=':', color=colors[i], markersize=20, markeredgewidth=5, linewidth=4)


            ax4.set_title(f'{file_types[i]} - Individual Joint distances', fontsize=18, fontweight='bold')
            # Add text annotations for distances between configurations            
            if i == 2:  # Add x-axis label only to the last subplot
                ax4.set_xlabel('Configuration Index', fontsize=20, fontweight='bold')
            ax4.set_ylabel('Normalized Value', fontsize=20, fontweight='bold')
            ax4.legend(fontsize=18)
            ax4.grid(True)
            ax4.tick_params(axis='both', labelsize=28, width=2)
            ax4.set_xlim(x_lim)

            
            # fig.suptitle(f'{file_type} - Individual Joint Distances', fontsize=20, fontweight='bold')

            # Create a new figure for this file type
            fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
            fig.suptitle(f'{file_types[i]} - Individual Joint Distances', fontsize=20, fontweight='bold')

            # Subplot 1: Keypoints distances
            axes[0].plot(kp_dist_direct_norm, label='Keypoints distances', marker='o', color=colors[i], markersize=12, markeredgewidth=4, linewidth=4)
            axes[0].set_title('Keypoints Distances')
            axes[0].set_ylabel('Normalized Value')
            axes[0].grid(True)
            axes[0].set_xlim(x_lim)            
            axes[0].tick_params(axis='both', labelsize=28, width=2)
            axes[0].legend(fontsize=18)

            # Subplot 2: Joint 1 distances
            axes[1].plot(j1_dist_norm, label='Joint 1 distances', marker='x', linestyle='--', color=colors[i], markersize=12, markeredgewidth=4, linewidth=4)
            axes[1].set_title('Joint 1 Distances')
            axes[1].set_ylabel('Normalized Value')
            axes[1].grid(True)
            axes[1].tick_params(axis='both', labelsize=28, width=2)
            axes[1].set_xlim(x_lim)
            axes[1].legend(fontsize=18)

            # Subplot 3: Joint 2 distances
            axes[2].plot(j2_dist_norm, label='Joint 2 distances', marker='s', linestyle='-.', color=colors[i], markersize=12, markeredgewidth=4, linewidth=4)
            axes[2].set_title('Joint 2 Distances')
            axes[2].set_ylabel('Normalized Value')
            axes[2].grid(True)
            axes[2].set_xlim(x_lim)
            axes[2].tick_params(axis='both', labelsize=28, width=2)
            axes[2].legend(fontsize=18)

            # Subplot 4: Joint 3 distances
            axes[3].plot(j3_dist_norm, label='Joint 3 distances', marker='d', linestyle=':', color=colors[i], markersize=12, markeredgewidth=4, linewidth=4)
            axes[3].set_title('Joint 3 Distances')
            axes[3].set_xlabel('Configuration Index')
            axes[3].set_ylabel('Normalized Value')
            axes[3].grid(True)
            axes[3].set_xlim(x_lim)
            axes[3].tick_params(axis='both', labelsize=28, width=2)
            axes[3].legend(fontsize=18)

            # Adjust layout and save
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for the title
            plot_filename = os.path.join(output_dir, f'exp_{exp}_{file_types[i].replace(" ", "_").lower()}_stacked_distances.svg')
            plt.savefig(plot_filename, format='svg')
            plt.close(fig)
            # print(f"Saved plot: {plot_filename}")

            # Plotting
            plt.figure(figsize=(10, 6))
            plt.plot(kp_dist_direct_norm, label='Normalized Keypoint Distances', marker='o', color='#5D3A9B')
            plt.plot(jt_dist_direct_norm, label='Normalized Joint Distances', marker='x', color='#D41159')
            plt.title(f'Experiment {exp} - {file_types[i]}')
            plt.xlabel('Configuration Index')
            plt.ylabel('Normalized Distance')
            plt.legend()
            plt.grid(True)

            # Save the plot
            plot_filename = os.path.join(output_dir, f'exp_{exp}_{file_types[i].replace(" ", "_").lower()}_no_obs.svg')
            plt.savefig(plot_filename, format='svg')
            plt.close()

            # print(f"Saved plot: {plot_filename}")

            
            # Compute the average distance
            # average_distance = np.mean(distances)

            # Print the results
        #     print(f"File: {file_path}")
        #     print(f"Number of configurations (excluding start): {number_of_configs}")
        #     print(f"Average keypoint distance between consecutive configurations: {average_distance:.6f}\n")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}\n")


        
    # Adjust layout
    # plt.tight_layout()

    # # Save the combined plot
    # merged_plot_filename = os.path.join(output_dir, f'exp_{exp}_merged_no_obs.svg')
    # plt.savefig(merged_plot_filename, format='svg')
    # plt.close()
            
    # Adjust layout and save the first figure
    fig1.tight_layout()
    # fig1.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1) 
    fig1.subplots_adjust(left=0.1) 
    abs_distances_plot_filename = os.path.join(output_dir, f'exp_{exp}_absolute_distances_no_obs.svg')
    fig1.savefig(abs_distances_plot_filename, format='svg')
    plt.close(fig1)
    # print(f"Saved normalized distances plot: {abs_distances_plot_filename}")

    # Adjust layout and save the second figure
    fig2.tight_layout()
    fig2.subplots_adjust(left=0.1)
    euc_distances_plot_filename = os.path.join(output_dir, f'exp_{exp}_euclidean_distances_no_obs.svg')
    fig2.savefig(euc_distances_plot_filename, format='svg')
    fig2.show()
    plt.close(fig2)
    # print(f"Saved normalized values plot: {euc_distances_plot_filename}")

    fig3.tight_layout()
    fig3.subplots_adjust(left=0.1)
    values_plot_filename = os.path.join(output_dir, f'exp_{exp}_normalized_values_no_obs.svg')
    fig3.savefig(values_plot_filename, format='svg')
    plt.close(fig3)
    # print(f"Saved normalized values plot: {values_plot_filename}")

    fig4.tight_layout()
    fig4.subplots_adjust(left=0.1)
    jt_values_plot_filename = os.path.join(output_dir, f'exp_{exp}_individual_jt_distances_with_obs.svg')
    fig4.savefig(jt_values_plot_filename, format='svg')
    plt.close(fig4)
    # print(f"Saved normalized values plot: {jt_values_plot_filename}")

from fractions import Fraction

# Two decimal numbers
num1 = 0.022403100775194
num2 = 3.13037686046512

# Convert the numbers to fractions
fraction_num1 = Fraction(num1).limit_denominator()
fraction_num2 = Fraction(num2).limit_denominator()

# Compute the ratio as a fraction
ratio = fraction_num1 / fraction_num2

# Print the ratio
# print(f"The ratio of {num1} to {num2} in fraction form is {ratio}")