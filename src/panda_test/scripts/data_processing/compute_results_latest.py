#!/usr/bin/env python3.8

import math
import numpy as np
import csv
# import os
import sys
# import rospy
from os.path import expanduser
from datetime import datetime

# import pandas as pd
# import numpy as np

# def compute_rise_time(data, sample_rate_hz, tolerance=0.02):
#     """
#     Compute rise time for each column in the DataFrame.
#     Rise time is the time taken for the signal to go from 10% to 90% of the final value.
#     """
#     rise_times = {}
#     time_per_step = 1 / sample_rate_hz  # Convert sample rate to time per step in seconds
    
#     for col in data.columns:
#         signal = data[col].values
#         final_value = signal[-1]
#         start_time = None
#         end_time = None
        
#         for i, value in enumerate(signal):
#             if start_time is None and value >= 0.1 * final_value:
#                 start_time = i
#             if start_time is not None and value >= 0.9 * final_value:
#                 end_time = i
#                 break

#         if start_time is not None and end_time is not None:
#             rise_times[col] = (end_time - start_time) * time_per_step
#         else:
#             rise_times[col] = np.nan  # If rise time cannot be determined
#     return rise_times

# def compute_settling_time(data, sample_rate_hz, tolerance=0.02):
#     """
#     Compute settling time for each column in the DataFrame.
#     Settling time is the time taken for the signal to remain within a certain tolerance band around the final value.
#     """
#     settling_times = {}
#     time_per_step = 1 / sample_rate_hz  # Convert sample rate to time per step in seconds
    
#     for col in data.columns:
#         signal = data[col].values
#         final_value = signal[-1]
#         within_tolerance = np.abs(signal - final_value) <= tolerance * np.abs(final_value)
        
#         settling_time = None
#         for i in range(len(within_tolerance) - 1, -1, -1):
#             if not within_tolerance[i]:
#                 settling_time = (i + 1) * time_per_step
#                 break
        
#         if settling_time is not None:
#             settling_times[col] = settling_time
#         else:
#             settling_times[col] = len(signal) * time_per_step  # If the signal is always within tolerance
#     return settling_times

# def compute_overshoot(data):
#     """
#     Compute overshoot for each column in the DataFrame.
#     Overshoot is the maximum peak value of the response curve minus the final value, as a percentage of the final value.
#     """
#     overshoots = {}
#     for col in data.columns:
#         signal = data[col].values
#         final_value = signal[-1]
#         max_value = np.max(signal)
        
#         if final_value != 0:
#             overshoot = ((max_value - final_value) / np.abs(final_value)) * 100
#         else:
#             overshoot = np.nan  # Avoid division by zero
        
#         overshoots[col] = overshoot
#     return overshoots

# # Load the CSV data
# # Replace 'file.csv' with your CSV file path
# file_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/euclidean/1/err.csv'
# data = pd.read_csv(file_path)

# # Drop the first column (assuming it is the 'current_goal' column)
# error_data = data.drop(columns=[data.columns[0]])
# sample_rate_hz = 10

# # Calculate rise time, settling time, and overshoot
# rise_times = compute_rise_time(error_data, sample_rate_hz)
# settling_times = compute_settling_time(error_data, sample_rate_hz)
# overshoots = compute_overshoot(error_data)

# # Combine results into a DataFrame
# results = pd.DataFrame({
#     'Feature': error_data.columns,
#     'Rise Time': rise_times.values(),
#     'Settling Time': settling_times.values(),
#     'Overshoot (%)': overshoots.values()
# })

# # Display the results
# print(results)

import pandas as pd
import numpy as np

def read_and_compute_total_error(file_path):
    # Read the CSV file, skipping the first 51 rows and the header
    df = pd.read_csv(file_path, skiprows=51)

    total_iterations = len(df)
    # Extract the initial values (first row after skipping)
    initial_values = df.iloc[0, 1:11].values  # Extracting columns 2 to 11 (x1, y1, x2, y2, ..., x5, y5)
    
    # Extract the final values (last row)
    final_values = df.iloc[-1, 1:11].values  # Extracting columns 2 to 11 (x1, y1, x2, y2, ..., x5, y5)
    
    # Calculate the errors for each feature from initial to final
    errors = final_values - initial_values
    
    # Calculate the error norms for pairs like (x1, y1), (x2, y2), ..., (x5, y5)
    error_norms = []
    for i in range(5):  # We have 5 pairs
        x_error = errors[2 * i]     # x1, x2, ..., x5
        y_error = errors[2 * i + 1] # y1, y2, ..., y5
        # Compute the norm for the pair (x_i, y_i)
        norm = np.linalg.norm([x_error, y_error])
        error_norms.append(norm)

    # Create a DataFrame to store the error values and their norms
    error_df = pd.DataFrame({
        'Feature Pair': [f'(x{i+1}, y{i+1})' for i in range(5)],
        'Error Norm': error_norms
    })

    # Display the DataFrame
    # print(error_df)

    return errors, total_iterations, error_df

def calculate_overshoot(data, max_errors):
    overshoot_results = []

    # Loop through each feature (pair of columns for x and y)
    for i in range(5):  # We have 5 features (x1, y1 to x5, y5)
        x_data = data[:, 2*i]  # x1, x2, x3, x4, x5
        y_data = data[:, 2*i + 1]  # y1, y2, y3, y4, y5

        # Compute overshoot for x
        z_cross_x = np.where(np.diff(np.signbit(x_data)))[0]

        overshoot_x = 0.0
        if z_cross_x.size > 0:
            if z_cross_x.size == 1:
                segment = np.abs(x_data[z_cross_x[0]:])
                overshoot_x = np.amax(segment)
            elif z_cross_x.size == 2:
                segment = np.abs(x_data[z_cross_x[0]:z_cross_x[1] + 1])
                overshoot_x = np.amax(segment)
            else:
                segment1 = np.abs(x_data[z_cross_x[0]:z_cross_x[1] + 1])
                segment2 = np.abs(x_data[z_cross_x[1]:z_cross_x[2] + 1])
                ov1 = np.amax(segment1)
                ov2 = np.amax(segment2)
                overshoot_x = min(ov1, ov2)

        # Compute overshoot for y
        z_cross_y = np.where(np.diff(np.signbit(y_data)))[0]

        overshoot_y = 0.0
        if z_cross_y.size > 0:
            if z_cross_y.size == 1:
                segment = np.abs(y_data[z_cross_y[0]:])
                overshoot_y = np.amax(segment)
            elif z_cross_y.size == 2:
                segment = np.abs(y_data[z_cross_y[0]:z_cross_y[1] + 1])
                overshoot_y = np.amax(segment)
            else:
                segment1 = np.abs(y_data[z_cross_y[0]:z_cross_y[1] + 1])
                segment2 = np.abs(y_data[z_cross_y[1]:z_cross_y[2] + 1])
                ov1 = np.amax(segment1)
                ov2 = np.amax(segment2)
                overshoot_y = min(ov1, ov2)

        # Compute average overshoot
        overshoot = math.sqrt(overshoot_x**2 + overshoot_y**2)
        
        # Normalize as a percentage of the initial error norm from the first CSV
        max_error_x = max_errors[2 * i]
        max_error_y = max_errors[2 * i + 1]
        error_norm = math.sqrt(max_error_x**2 + max_error_y**2)
        
        if error_norm != 0:
            overshoot_percentage = (overshoot / error_norm) * 100
        else:
            overshoot_percentage = 0.0

        overshoot_results.append(overshoot_percentage)

    return overshoot_results

def calculate_settling_time(error_data, steady_state_values, tolerance=0.02, control_rate_hz=10):
    """
    Calculate the settling time for each feature.
    
    Parameters:
    - error_data: np.array, shape (n_samples, n_features*2)
    - steady_state_values: List of steady-state values for each feature
    - tolerance: Percentage tolerance band (default is 0.02 for ±2%)
    - control_rate_hz: Control frequency in Hz (default is 10 Hz)
    
    Returns:
    - List of settling times for each feature
    """
    settling_times = []

    time = 0.0
    bound = 0.02 * max_errors
    flag = False
    st_end = 0

    for i in range(iterations):
        if data[i] <= bound and not flag:
            st_end = i
            flag = True
    time = (st_end+1)/control_rate_hz
    return time

# Load your data from CSV (example: 'control_data.csv')
# Replace 'control_data.csv' with your actual file path
first_csv_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_no_obs_v1/1/cp.csv'
second_csv_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_no_obs_v1/1/err.csv'

# Step 1: Read the first CSV and compute total error
max_errors, iterations, error_df = read_and_compute_total_error(first_csv_path)
# print(max_errors, iterations, error_df)
print(f"Errors: {max_errors} \n total_data_points: {iterations} \n Error_in_dict: {error_df}")


# # Step 2: Read the second CSV (error data) skipping the first row
# df = pd.read_csv(second_csv_path, header=0, skiprows=1)
# # Extract values from the 2nd to 11th columns
# error_data = df.iloc[:, 1:11].values

# print(error_data)
# # Step 3: Calculate overshoot for each control feature
# overshoot_percentages = calculate_overshoot(error_data, max_errors)
# # Display results
# for i, overshoot in enumerate(overshoot_percentages, 1):
#     print(f"Feature {i} Overshoot (%): {overshoot:.2f}")

# settling_times = calculate_settling_time(error_data, max_errors, tolerance=0.02, control_rate_hz=10)

# # Display results
# for i, settling_time in enumerate(settling_times, 1):
#     print(f"Feature {i} Settling Time (s): {settling_time:.2f} seconds")

