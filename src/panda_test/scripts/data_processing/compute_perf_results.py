#!/usr/bin/env python3.8

import pandas as pd
import numpy as np
import math
from collections import Counter
import os

def compute_feature_based_percentage_overshoot(df):
    
    # Initialize an empty dictionary to store percentage overshoot values for each feature pair
    feature_percentage_overshoot = {}

    # Identify the last goal value
    last_goal = df.iloc[:, 0].max()  # First column (current_goal) is at index 0

    # Filter the data for only the rows corresponding to the last goal
    df_last_goal = df[df.iloc[:, 0] == last_goal]

    initial_values_df = df.groupby(df.iloc[:, 0]).first() # Group by goal values and extract the first row

    # Iterate through each feature pair (e.g., Err_cp3_x, Err_cp3_y, etc.)
    for i in range(1, df_last_goal.shape[1], 2):
        feature_x = df_last_goal.iloc[:, i].values
        feature_y = df_last_goal.iloc[:, i + 1].values

        initial_x_values = np.abs(initial_values_df.iloc[:, i][initial_values_df.index < last_goal].values)
        initial_y_values = np.abs(initial_values_df.iloc[:, i + 1][initial_values_df.index < last_goal].values)

        initial_x = np.append(initial_x_values, np.abs(feature_x[0]))
        initial_y = np.append(initial_y_values, np.abs(feature_y[0]))

        initial_x_sum = initial_x.sum()
        initial_y_sum = initial_y.sum()

        # print(feature_x, initial_x)
        # print(feature_y, initial_y)

        # Detect zero-crossing indices using signbit changes
        def max_after_zero_crossing_with_signbit(values):
            sign_changes = np.diff(np.signbit(values)).astype(int)  # Detect sign changes using np.signbit
            zero_crossing_indices = np.where(sign_changes != 0)[0]  # Get indices where sign change occurs

            # If no zero-crossing, return None
            if len(zero_crossing_indices) == 0:
                return None
            
            # print(zero_crossing_indices)

            # Segment the values based on zero-crossing points
            segment_max_values = []
            for start_idx, end_idx in zip(zero_crossing_indices[:-1], zero_crossing_indices[1:]):
                segment_max = np.max(np.abs(values[start_idx + 1:end_idx + 1]))  # Max in each segment
                segment_max_values.append(segment_max)

            # Include the last segment after the final zero-crossing
            segment_max_values.append(np.max(np.abs(values[zero_crossing_indices[-1] + 1:])))

            # Return the maximum value of the last segment
            return segment_max_values[-1]

        # Calculate overshoot for x and y features using zero-crossing logic
        max_overshoot_x = max_after_zero_crossing_with_signbit(feature_x)
        max_overshoot_y = max_after_zero_crossing_with_signbit(feature_y)

        # print(max_overshoot_x)
        # print(max_overshoot_y)

        print(f"Max Overshoot X: {max_overshoot_x}, Max Overshoot Y: {max_overshoot_y}")

        # Calculate percentage overshoot if both values are not None
        if max_overshoot_x is not None and max_overshoot_y is not None:
            # Calculate the Euclidean norm of the overshoot
            overshoot_norm = math.sqrt(max_overshoot_x**2 + max_overshoot_y**2)

            # Calculate the initial magnitude using the cumulative initial values
            initial_magnitude = math.sqrt(initial_x_sum**2 + initial_y_sum**2)

            # Calculate percentage overshoot
            percentage_overshoot = (overshoot_norm / initial_magnitude) * 100 if initial_magnitude != 0 else None
            print(f"Percentage Overshoot for Feature Pair {i//2 + 1}: {percentage_overshoot}")

        else:
            # Calculate the steady-state error if no zero-crossing is found
            # Use the last 10 values of each feature and find the most frequent value
            last_10_x = feature_x[-10:]  # Last 10 values of feature_x
            last_10_y = feature_y[-10:]  # Last 10 values of feature_y

            # Find the most common value (steady-state value) using Counter
            steady_state_x = Counter(last_10_x).most_common(1)[0][0]
            steady_state_y = Counter(last_10_y).most_common(1)[0][0]

            print(f"Steady-State Error for X: {steady_state_x}, Steady-State Error for Y: {steady_state_y}")

            # Calculate the Euclidean norm of the steady-state error
            steady_state_error_norm = math.sqrt(steady_state_x**2 + steady_state_y**2)

            # Calculate the initial magnitude using the cumulative initial values
            initial_magnitude = math.sqrt(initial_x_sum**2 + initial_y_sum**2)

            # Calculate percentage steady-state error
            percentage_steady_state_error = (steady_state_error_norm / initial_magnitude) * 100 if initial_magnitude != 0 else None
            print(f"Percentage Steady-State Error for Feature Pair {i//2 + 1}: {percentage_steady_state_error}")

            # Store steady-state error percentage instead of overshoot
            percentage_overshoot = percentage_steady_state_error

        # Store the percentage overshoot value for the current feature pair
        feature_number = (i // 2) + 1
        feature_percentage_overshoot[f'Feature {feature_number} Overshoot (%)'] = percentage_overshoot

       
    # Convert the metrics to a DataFrame for easy visualization
    percentage_overshoot_df = pd.DataFrame(list(feature_percentage_overshoot.items()), columns=['Feature Pair', 'Percentage Overshoot'])   


    return percentage_overshoot_df

# Function to compute settling time for each feature based on a 2% error bound
def compute_settling_time(df, error_bound_factor=0.04):
    # Initialize a dictionary to store settling time for each feature pair

    feature_settling_time = {}

    # Identify the last goal value
    last_goal = df.iloc[:, 0].max()  # First column (current_goal) is at index 0

    # Filter the data for only the rows corresponding to the last goal
    df_last_goal = df[df.iloc[:, 0] == last_goal]

    # Get the indices of the rows corresponding to the last goal
    last_goal_indices = df.index[df.iloc[:, 0] == last_goal].tolist()

    # print(last_goal_indices)

    # Extract the initial values for each goal
    initial_values_df = df.groupby(df.iloc[:, 0]).first() # Group by goal values and extract the first row

    # print("Initial Values", initial_values_df)

    # Calculate settling time for each feature pair (e.g., Err_cp3_x, Err_cp3_y)
    for i in range(1, df_last_goal.shape[1], 2):
        # print(i)
        feature_x_full = df.iloc[:, i].values  # Full data for feature X
        feature_y_full = df.iloc[:, i + 1].values  # Full data for feature Y
        # print(feature_x_full, feature_y_full)

        feature_x = df_last_goal.iloc[:, i].values
        feature_y = df_last_goal.iloc[:, i + 1].values

        initial_x_values = np.abs(initial_values_df.iloc[:, i][initial_values_df.index < last_goal].values)
        initial_y_values = np.abs(initial_values_df.iloc[:, i + 1][initial_values_df.index < last_goal].values)

        initial_x = np.append(initial_x_values, np.abs(feature_x[0]))
        initial_y = np.append(initial_y_values, np.abs(feature_y[0]))

        initial_x_sum = initial_x.sum()
        initial_y_sum = initial_y.sum()

        # print("Cumulative x", initial_x_sum)
        # print("Cumulative Y", initial_y_sum)
        # Calculate the 2% error bound
        cumulative_initial_error = math.sqrt(initial_x_sum**2 + initial_y_sum**2)
        error_bound = cumulative_initial_error * error_bound_factor

        # print("Error bound", error_bound)
        # print("Error norm", cumulative_initial_error)

        # Calculate the settling time index
        settling_idx = []
        # for idx in range(len(feature_x_full)):
        for idx in last_goal_indices:
            # print(idx)
            # Calculate the Euclidean distance from the initial value
            settling_error = math.sqrt(feature_x_full[idx]**2 + feature_y_full[idx]**2)
            if settling_error <= error_bound:
                print("Settling error in case within  bound", settling_error)
                # print(error_bound)
                settling_idx.append(idx)
            else:
                print("Error not within bound")

        settling_idx = min(settling_idx)
        # print(settling_idx)
        # Calculate the settling time in seconds (iteration divided by 10 to get seconds)
        if settling_idx is not None:
            settling_time = (settling_idx)/10.0  # Assuming each iteration is 0.1 seconds
        else:
            settling_time = None  # If never settles
        # Store the settling time for the current feature pair
        feature_number = (i // 2) + 1
        feature_settling_time[f'Feature {feature_number} Settling Time (s)'] = settling_time

    # Convert to DataFrame for easier viewing
    settling_time_df =  pd.DataFrame(list(feature_settling_time.items()), columns=['Feature Pair', 'Settling Time (s)'])

    return settling_time_df

def compute_rise_time(df, lower_bound=0.1, upper_bound=0.9):
    feature_rise_time = {}
    last_goal = df.iloc[:, 0].max()
    df_last_goal = df[df.iloc[:, 0] == last_goal]
    last_goal_indices = df.index[df.iloc[:, 0] == last_goal].tolist()
    initial_values_df = df.groupby(df.iloc[:, 0]).first()

    for i in range(1, df_last_goal.shape[1], 2):
        feature_x_full = df.iloc[:, i].values
        feature_y_full = df.iloc[:, i + 1].values
        initial_x_values = np.abs(initial_values_df.iloc[:, i][initial_values_df.index < last_goal].values)
        initial_y_values = np.abs(initial_values_df.iloc[:, i + 1][initial_values_df.index < last_goal].values)

        initial_x = np.append(initial_x_values, np.abs(df_last_goal.iloc[0, i]))
        initial_y = np.append(initial_y_values, np.abs(df_last_goal.iloc[0, i + 1]))
        initial_x_sum = initial_x.sum()
        initial_y_sum = initial_y.sum()
        cumulative_initial_error = math.sqrt(initial_x_sum**2 + initial_y_sum**2)

        lower_threshold = cumulative_initial_error * lower_bound
        upper_threshold = cumulative_initial_error * upper_bound

        # print(lower_threshold, upper_threshold)

        # Search for the lower index (10%) in the entire trajectory
        lower_idx = None
        for idx in range(len(feature_x_full)):
            rise_error = math.sqrt(feature_x_full[idx]**2 + feature_y_full[idx]**2)            
            if rise_error <= lower_threshold:
                # print(rise_error)
                lower_idx = idx
                # print(lower_idx)
                break  # Found the first point reaching the lower threshold

        # Ensure the `lower_idx` is valid before proceeding
        if lower_idx is None:
            feature_rise_time[f'Feature {(i // 2) + 1} Rise Time (s)'] = None
            continue

        # Search for the upper index (90%) within the `last_goal_indices` to ensure it is captured in the last goal
        upper_idx = None
        for idx in last_goal_indices:
            rise_error = math.sqrt(feature_x_full[idx]**2 + feature_y_full[idx]**2)
            if rise_error <= upper_threshold:
                # print(rise_error)
                upper_idx = idx
                break  # Found the first point reaching the upper threshold

        # Calculate rise time if both indices are found
        if lower_idx is not None and upper_idx is not None:
            rise_time = (upper_idx - lower_idx)/10.0  # Convert to seconds assuming each iteration is 0.1 seconds
        else:
            rise_time = None  # If never reaches the upper threshold

        feature_number = (i // 2) + 1
        feature_rise_time[f'Feature {feature_number} Rise Time (s)'] = rise_time

    rise_time_df = pd.DataFrame(list(feature_rise_time.items()), columns=['Feature Pair', 'Rise Time (s)'])

    return rise_time_df

# Main function to compute metrics
def compute_metrics(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, skiprows=1)
    # print(len(df))
    
    # Compute percentage overshoot
    percentage_overshoot_results = compute_feature_based_percentage_overshoot(df)

    # Compute settling time
    settling_time_results = compute_settling_time(df)

    rise_time_results = compute_rise_time(df)

    execution_time = len(df)/10.0

    return percentage_overshoot_results, settling_time_results, rise_time_results, execution_time

def compute_metrics_for_multiple_experiments(base_path, exps_list):
    # Create empty lists to store consolidated results for each experiment
    consolidated_results = []

    # Iterate over the list of experiments
    for exp_id in exps_list:
        # Create the csv_path dynamically based on experiment ID
        csv_path = os.path.join(base_path, str(exp_id), 'err.csv')

        # Check if the file exists before processing
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}. Skipping experiment {exp_id}.")
            continue

        print(f"Processing Experiment: {exp_id}")

        # Calculate metrics for the current experiment
        percentage_overshoot, settling_time, rise_time, execution_time = compute_metrics(csv_path)

        # Initialize a dictionary to hold consolidated results for this experiment
        experiment_results = {'Experiment ID': exp_id, 'Execution Time (s)': execution_time}

        # Extract values for rise time, settling time, and overshoot for each feature
        for index, row in rise_time.iterrows():
            experiment_results[f'Rise Time Feature {index + 1} (s)'] = row['Rise Time (s)']
        for index, row in settling_time.iterrows():
            experiment_results[f'Settling Time Feature {index + 1} (s)'] = row['Settling Time (s)']
        for index, row in percentage_overshoot.iterrows():
            experiment_results[f'Overshoot Feature {index + 1} (%)'] = row['Percentage Overshoot']

        # Append this experiment's results to the consolidated list
        consolidated_results.append(experiment_results)

    # Convert the consolidated list to a DataFrame
    consolidated_results_df = pd.DataFrame(consolidated_results)

    # Save the consolidated results to a CSV file
    consolidated_results_file = os.path.join(base_path, 'consolidated_experiment_results.csv')
    consolidated_results_df.to_csv(consolidated_results_file, index=False)
    print(f"Consolidated results saved to: {consolidated_results_file}")

    return consolidated_results_df

exps_list = [9,10,17,19,20]

# Example of how to call the function:
# csv_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_no_obs_v1/1/err.csv'
base_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/euclidean/a_star_latest_with_obs/'

# Calculate metrics
# percentage_overshoot, settling_time, rise_time, execution_time = compute_metrics(csv_path)

consolidated_results = compute_metrics_for_multiple_experiments(base_path, exps_list)


# Save to CSV
# percentage_overshoot.to_csv('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_no_obs_v1/1/percentage_overshoot_results.csv', index=False)
# settling_time.to_csv('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_no_obs_v1/1/settling_time_results.csv', index=False)

# Display results
# print("Percentage Overshoot Results:\n", percentage_overshoot)
# print("Settling Time Results:\n", settling_time)
# print("Rise Time Results:\n", rise_time)
# print("Execution Time Results:\n", execution_time)

