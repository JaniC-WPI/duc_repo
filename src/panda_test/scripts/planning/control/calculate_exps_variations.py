
#!/usr/bin/env python3.8

import pandas as pd
import numpy as np

def calculate_jerk(joint_positions):
    """
    Calculates the jerk (rate of change of acceleration) for a given sequence of joint positions.
    
    Args:
    - joint_positions (pd.Series): A pandas series of joint positions (e.g., Joint 1, Joint 2, Joint 3).
    
    Returns:
    - jerk: A measure of the smoothness of the trajectory.
    """
    velocity = np.gradient(joint_positions)  # First derivative
    acceleration = np.gradient(velocity)     # Second derivative (acceleration)
    jerk = np.gradient(acceleration)         # Third derivative (jerk)
    
    return np.linalg.norm(jerk)

def calculate_total_variation(joint_positions):
    """
    Calculates the total variation for a given sequence of joint positions.
    
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
    Calculates the total Euclidean distance between consecutive joint configurations.
    
    Args:
    - joint1, joint2, joint3 (pd.Series): Joint angle series for each joint.
    
    Returns:
    - total_distances: The total Euclidean distance between consecutive joint configurations.
    """
    distances = []
    for i in range(len(joint1) - 1):
        dist = np.linalg.norm([joint1.iloc[i+1] - joint1.iloc[i],
                               joint2.iloc[i+1] - joint2.iloc[i],
                               joint3.iloc[i+1] - joint3.iloc[i]])
        distances.append(dist)
    
    total_distances = np.sum(distances)
    return total_distances

def process_file(file_path):
    df = pd.read_csv(file_path)
    joint1 = df['Joint 1']
    joint2 = df['Joint 2']
    joint3 = df['Joint 3']

    # Calculate variations
    variation_joint1 = calculate_total_variation(joint1)
    variation_joint2 = calculate_total_variation(joint2)
    variation_joint3 = calculate_total_variation(joint3)

    # Calculate total distances
    total_distances = calculate_average_distance(joint1, joint2, joint3)

    return variation_joint1, variation_joint2, variation_joint3, total_distances

def main():
    # Paths to the folders
    base_path_custom = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/no_obs/'
    base_path_euclidean = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/no_obs/'
    base_path_ground_truth = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/no_obs/'

    results = []

    # List of folders to process
    exps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # Process files for each specified folder in exps
    for i in exps:
        file_name = f'{i}/save_distances.csv'
        
        # Paths to each CSV file
        custom_path = f'{base_path_custom}/{file_name}'
        euclidean_path = f'{base_path_euclidean}/{file_name}'
        ground_truth_path = f'{base_path_ground_truth}/{file_name}'

        # Initialize placeholders for variations and distances
        variation_custom = (np.nan, np.nan, np.nan, np.nan)
        variation_euclidean = (np.nan, np.nan, np.nan, np.nan)
        variation_ground_truth = (np.nan, np.nan, np.nan, np.nan)

        # Process each file
        try:
            variation_custom = process_file(custom_path)
            variation_euclidean = process_file(euclidean_path)
            variation_ground_truth = process_file(ground_truth_path)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

        # Append the results
        result_row = [
            file_name,  # Filename
            *variation_ground_truth,  # Joint variations for Ground Truth
            *variation_custom,  # Joint variations for Custom
            *variation_euclidean  # Joint variations for Euclidean
        ]

        print(f"Processed {file_name}: {len(result_row)} columns")  # Debug: Check row length

        results.append(result_row)

    # Create a DataFrame to store the results
    columns = ['File Name', 'Ground Truth Var 1', 'Ground Truth Var 2', 'Ground Truth Var 3', 'Ground Truth Total Distance',
               'Custom Var 1', 'Custom Var 2', 'Custom Var 3', 'Custom Total Distance',
               'Euclidean Var 1', 'Euclidean Var 2', 'Euclidean Var 3', 'Euclidean Total Distance']
    
    results_df = pd.DataFrame(results, columns=columns)

    # Save the results to a CSV file
    output_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_plots/no_obs/joint_variations_summary.csv'
    results_df.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()