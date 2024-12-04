import os
import pandas as pd

# Base directory where folders containing save_distances.csv files are located
base_dir = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/with_obs/'

# List of experiments (folder names)
exps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Output CSV file to store results
output_file = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/with_obs/joint_distance_summary.csv'

# List to store experiment results
results = []

# Iterate over each folder in the exps list
for exp in exps:
    # Path to the specific experiment folder
    folder_path = os.path.join(base_dir, str(exp))
    
    # Path to the save_distances.csv file
    csv_file_path = os.path.join(folder_path, 'save_distances.csv')
    
    # Check if save_distances.csv exists in the folder
    if os.path.exists(csv_file_path):
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Check if 'Distance to next Joint Angles' column exists
        if 'Distance to next Joint Angles' in df.columns:
            # Sum the values of the 'Distance to next Joint Angles' column
            total_joint_distance = df['Distance to next Joint Angles'].sum()
            
            # Append the experiment (folder name) and total_joint_distance to results
            results.append([exp, total_joint_distance])

# Convert results to a DataFrame
results_df = pd.DataFrame(results, columns=['Experiment', 'Total Joint Distance'])

# Save results to CSV
results_df.to_csv(output_file, index=False)

print(f"Joint distances have been saved to {output_file}")