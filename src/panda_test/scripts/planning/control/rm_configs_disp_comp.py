#!/usr/bin/env python3
import json
import numpy as np
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pos_regression_control import PosRegModel  # Ensure this is correctly imported

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load keypoints from JSON files in a given directory
def load_keypoints_from_json(directory):
    configurations = []
    configuration_ids = []
    for filename in os.listdir(directory):
        if filename.endswith('.json') and not filename.endswith('_joint_angles.json') and not filename.endswith('_vel.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]
                           
                # Check if any keypoint's y component is greater than 390
                if any(kp[1] > 390 for kp in keypoints[2:]):  # Start checking from the third keypoint
                    print(f"Skipping configuration {data['id']} due to a keypoint (from third onward) with y > 390.")
                    continue
                
                configurations.append(np.array(keypoints))
                configuration_ids.append(data['id'])

    print(f"Number of valid configurations: {len(configurations)}")
    return configurations, configuration_ids

# Load joint angles from JSON files in a given directory
def load_joint_angles_from_json(directory):
    joint_angles_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('_joint_angles.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                joint_angles_dict[data['id']] = np.array(data['joint_angles'])
    return joint_angles_dict

# Load the trained model for inference
def load_model_for_inference(model_path):    
    model = PosRegModel(18)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # Set the model to inference mode
    return model

# Predict joint displacement using the custom model
def predict_custom_distance(current_config, next_config, model):
    start_kp_flat = torch.tensor(current_config.flatten(), dtype=torch.float).unsqueeze(0).to(device)
    next_kp_flat = torch.tensor(next_config.flatten(), dtype=torch.float).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(start_kp_flat, next_kp_flat).squeeze(0).cpu().numpy()  # Remove batch dimension for output

    return output

# Function to calculate actual joint displacement between two configurations
def calculate_actual_joint_displacement(current_joint_angles, next_joint_angles):
    return next_joint_angles - current_joint_angles

# Function to skip configurations and gather the corresponding joint angles
def skip_configurations_and_gather_joint_angles(configurations, configuration_ids, joint_angles_dict, skip_step=5, start=1, end=13000):
    """
    Skip configurations and gather the corresponding joint angles.

    Args:
    - configurations: List of configurations (keypoints).
    - configuration_ids: List of configuration IDs corresponding to the configurations.
    - joint_angles_dict: Dictionary containing actual joint angles keyed by configuration ID.
    - skip_step: The step size for skipping configurations.
    - start: The starting index for skipping.
    - end: The ending index for skipping.

    Returns:
    - skipped_configs: List of skipped configurations.
    - skipped_ids: List of skipped configuration IDs.
    - skipped_joint_angles: List of joint angles for the skipped configurations.
    """
    skipped_configs = configurations[start:end:skip_step]
    skipped_ids = configuration_ids[start:end:skip_step]
    skipped_joint_angles = [joint_angles_dict[config_id] for config_id in skipped_ids]

    print(len(skipped_configs))

    return skipped_configs, skipped_ids, skipped_joint_angles

# Function to calculate actual and predicted joint displacements between consecutive configurations in the skipped list
def compare_displacements_in_skipped_list(skipped_configs, skipped_joint_angles, model):
    """
    Calculate actual and predicted joint displacements between consecutive configurations in the skipped list.

    Args:
    - skipped_configs: List of skipped configurations.
    - skipped_joint_angles: List of joint angles for the skipped configurations.
    - model: The trained model used for predicting joint displacements.

    Returns:
    - actual_displacements_list: List of actual joint displacements.
    - predicted_displacements_list: List of predicted joint displacements.
    """
    actual_displacements_list = []
    predicted_displacements_list = []
    
    for i in range(len(skipped_configs) - 1):
        current_config = skipped_configs[i]
        next_config = skipped_configs[i + 1]
        
        # Predict joint displacement using the model
        predicted_displacement = predict_custom_distance(current_config, next_config, model)
        predicted_displacements_list.append(predicted_displacement)
        
        # Calculate actual joint displacement
        current_joint_angles = skipped_joint_angles[i]
        next_joint_angles = skipped_joint_angles[i + 1]
        actual_displacement = calculate_actual_joint_displacement(current_joint_angles, next_joint_angles)
        actual_displacements_list.append(actual_displacement)
    
    return actual_displacements_list, predicted_displacements_list

# Plot histograms for side-by-side comparison of actual and predicted joint displacements
def plot_joint_displacement_side_by_side_histograms(actual_displacements_list, predicted_displacements_list):
    actual_displacements = np.array(actual_displacements_list)
    predicted_displacements = np.array(predicted_displacements_list)
    
    # Create a DataFrame for Seaborn visualizations
    displacement_data_df = pd.DataFrame({
        'Joint 1 Actual': actual_displacements[:, 0],
        'Joint 1 Predicted': predicted_displacements[:, 0],
        'Joint 2 Actual': actual_displacements[:, 1],
        'Joint 2 Predicted': predicted_displacements[:, 1],
        'Joint 3 Actual': actual_displacements[:, 2],
        'Joint 3 Predicted': predicted_displacements[:, 2]
    })

    # Plot side-by-side histograms for each joint with consistent bin width
    plt.figure(figsize=(18, 15))
    for i, joint in enumerate(['Joint 1', 'Joint 2', 'Joint 3'], 1):
        # Calculate the common range for actual and predicted displacements and set bin width
        min_value = min(displacement_data_df[f'{joint} Actual'].min(), displacement_data_df[f'{joint} Predicted'].min())
        max_value = max(displacement_data_df[f'{joint} Actual'].max(), displacement_data_df[f'{joint} Predicted'].max())
        bin_width = 0.1  # Set your desired bin width
        bins = np.arange(min_value, max_value + bin_width, bin_width)

        # Actual joint displacement histogram
        plt.subplot(3, 2, 2 * i - 1)
        sns.histplot(displacement_data_df[f'{joint} Actual'], bins=bins, kde=False, color='blue', alpha=0.6)
        plt.title(f'Actual {joint} Displacements', fontsize=14, fontweight='bold')
        plt.xlabel('Displacement', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')

        plt.xlim(-2.7,2.7)
        plt.ylim(0, 850)

        # Predicted joint displacement histogram
        plt.subplot(3, 2, 2 * i)
        sns.histplot(displacement_data_df[f'{joint} Predicted'], bins=bins, kde=False, color='red', alpha=0.6)
        plt.title(f'Predicted {joint} Displacements', fontsize=14, fontweight='bold')
        plt.xlabel('Displacement', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')

        plt.xlim(-2.7,2.7)
        plt.ylim(0, 850)

    plt.tight_layout()
    plt.show()

# Plot histograms for each joint in separate figures
def plot_joint_displacement_individual_histograms(actual_displacements_list, predicted_displacements_list):
    actual_displacements = np.array(actual_displacements_list)
    predicted_displacements = np.array(predicted_displacements_list)
    
    # Create a DataFrame for Seaborn visualizations
    displacement_data_df = pd.DataFrame({
        'Joint 1 Actual': actual_displacements[:, 0],
        'Joint 1 Predicted': predicted_displacements[:, 0],
        'Joint 2 Actual': actual_displacements[:, 1],
        'Joint 2 Predicted': predicted_displacements[:, 1],
        'Joint 3 Actual': actual_displacements[:, 2],
        'Joint 3 Predicted': predicted_displacements[:, 2]
    })

    # Plot histograms for each joint in separate figures
    for i, joint in enumerate(['Joint 1', 'Joint 2', 'Joint 3'], 1):
        # Calculate the common range for actual and predicted displacements and set bin width
        min_value = min(displacement_data_df[f'{joint} Actual'].min(), displacement_data_df[f'{joint} Predicted'].min())
        max_value = max(displacement_data_df[f'{joint} Actual'].max(), displacement_data_df[f'{joint} Predicted'].max())
        bin_width = 0.1  # Set your desired bin width
        bins = np.arange(min_value, max_value + bin_width, bin_width)

        # Create a new figure for each joint
        plt.figure(figsize=(12, 6))

        # Actual joint displacement histogram
        plt.subplot(1, 2, 1)
        sns.histplot(displacement_data_df[f'{joint} Actual'], bins=bins, kde=False, color='#40B0A6', alpha=0.9)
        # plt.title(f'Actual {joint} Displacements', fontsize=14, fontweight='bold')
        # plt.xlabel('Displacement', fontsize=12, fontweight='bold')
        # plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.xlabel('')
        plt.ylabel('')
        plt.xlim(-2.7, 2.7)
        plt.ylim(0, 600)

        plt.xticks(fontsize=18, weight='bold')
        plt.yticks(fontsize=18, weight='bold')

        # Predicted joint displacement histogram
        plt.subplot(1, 2, 2)
        sns.histplot(displacement_data_df[f'{joint} Predicted'], bins=bins, kde=False, color='#5D3A9B', alpha=0.9)
        # plt.title(f'Predicted {joint} Displacements', fontsize=14, fontweight='bold')
        # plt.xlabel('Displacement', fontsize=12, fontweight='bold')
        # plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.xlabel('')
        plt.ylabel('')
        plt.xlim(-2.7, 2.7)
        plt.ylim(0, 600)

        plt.xticks(fontsize=18, weight='bold')
        plt.yticks(fontsize=18, weight='bold')

        # Adjust the layout and show the plot for this joint
        plt.tight_layout()
        plt.show()

# Function to calculate mean absolute error (MAE) and root mean squared error (RMSE)
def calculate_error_metrics(actual_displacements, predicted_displacements):
    actual_displacements = np.array(actual_displacements)
    predicted_displacements = np.array(predicted_displacements)

    mae = np.mean(np.abs(actual_displacements - predicted_displacements), axis=0)
    rmse = np.sqrt(np.mean((actual_displacements - predicted_displacements) ** 2, axis=0))
    
    return mae, rmse

# Function to plot scatter plots comparing actual vs. predicted displacements
def plot_actual_vs_predicted_scatter(actual_displacements_list, predicted_displacements_list):
    actual_displacements = np.array(actual_displacements_list)
    predicted_displacements = np.array(predicted_displacements_list)

    # Create a DataFrame for Seaborn visualizations
    displacement_data_df = pd.DataFrame({
        'Joint 1 Actual': actual_displacements[:, 0],
        'Joint 1 Predicted': predicted_displacements[:, 0],
        'Joint 2 Actual': actual_displacements[:, 1],
        'Joint 2 Predicted': predicted_displacements[:, 1],
        'Joint 3 Actual': actual_displacements[:, 2],
        'Joint 3 Predicted': predicted_displacements[:, 2]
    })

    # Plot scatter plots for each joint
    for joint in ['Joint 1', 'Joint 2', 'Joint 3']:
        plt.figure(figsize=(8, 8))
        sns.scatterplot(
            x=displacement_data_df[f'{joint} Actual'],
            y=displacement_data_df[f'{joint} Predicted'],
            alpha=0.6, color='#1f77b4'
        )
        plt.plot(
            [displacement_data_df[f'{joint} Actual'].min(), displacement_data_df[f'{joint} Actual'].max()],
            [displacement_data_df[f'{joint} Actual'].min(), displacement_data_df[f'{joint} Actual'].max()],
            color='red', linestyle='--', label='y = x'
        )
        plt.title(f'Actual vs. Predicted {joint} Displacements', fontsize=14, fontweight='bold')
        plt.xlabel('Actual Displacement', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted Displacement', fontsize=12, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()

# Function to calculate the error between actual and predicted joint displacements
def calculate_displacement_error(actual_displacements_list, predicted_displacements_list):
    actual_displacements = np.array(actual_displacements_list)
    predicted_displacements = np.array(predicted_displacements_list)
    error_displacements = actual_displacements - predicted_displacements
    return error_displacements

# Plot histograms for ground truth vs. predicted error comparison
def plot_displacement_error_histograms(error_displacements):
    # Create a DataFrame for Seaborn visualizations
    error_data_df = pd.DataFrame({
        'Joint 1 Error': error_displacements[:, 0],
        'Joint 2 Error': error_displacements[:, 1],
        'Joint 3 Error': error_displacements[:, 2]
    })

    # Plot histograms for each joint's error in separate figures
    for i, joint in enumerate(['Joint 1', 'Joint 2', 'Joint 3'], 1):
        # Calculate the common range for error displacements and set bin width
        min_value = error_data_df[f'{joint} Error'].min()
        max_value = error_data_df[f'{joint} Error'].max()
        bin_width = 0.1  # Set your desired bin width
        bins = np.arange(min_value, max_value + bin_width, bin_width)

        # Create a new figure for each joint's error
        plt.figure(figsize=(12, 6))

        # Plot the error histogram
        sns.histplot(error_data_df[f'{joint} Error'], bins=bins, kde=False, color='#E66100', alpha=0.9)
        # plt.title(f'{joint} Displacement Error', fontsize=14, fontweight='bold')
        # plt.xlabel('Error (Actual - Predicted)', fontsize=12, fontweight='bold')
        # plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.axvline(0, color='black', linestyle='--')  # Add a vertical line at 0 for reference

        plt.xlabel('')
        plt.ylabel('')
        plt.xlim(-2.5, 2.5)
        plt.ylim(0, 250)

        plt.xticks(fontsize=18, weight='bold')
        plt.yticks(fontsize=18, weight='bold')

        # Adjust the layout and show the plot for this joint
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged/'
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b128_e400_v32.pth'

    # Load configurations and joint angles
    configurations, configuration_ids = load_keypoints_from_json(directory)
    joint_angles_dict = load_joint_angles_from_json(directory)

    # Load the trained model
    model = load_model_for_inference(model_path)

    # Set parameters for skipping configurations
    skip_step = 10
    start_index = 1
    end_index = 28000

    # Gather skipped configurations and joint angles
    skipped_configs, skipped_ids, skipped_joint_angles = skip_configurations_and_gather_joint_angles(
        configurations, configuration_ids, joint_angles_dict, skip_step, start_index, end_index
    )

    # Calculate actual and predicted joint displacements
    actual_displacements_list, predicted_displacements_list = compare_displacements_in_skipped_list(
        skipped_configs, skipped_joint_angles, model
    )

    # Plot histograms for comparison
    plot_joint_displacement_side_by_side_histograms(actual_displacements_list, predicted_displacements_list)

    plot_joint_displacement_individual_histograms(actual_displacements_list, predicted_displacements_list)

    # Calculate error metrics
    mae, rmse = calculate_error_metrics(actual_displacements_list, predicted_displacements_list)
    print("Mean Absolute Error (MAE) for each joint:", mae)
    print("Root Mean Squared Error (RMSE) for each joint:", rmse)

    # Plot scatter plots for comparison
    plot_actual_vs_predicted_scatter(actual_displacements_list, predicted_displacements_list)

    error_displacements = calculate_displacement_error(actual_displacements_list, predicted_displacements_list)

    # Plot histograms for error comparison
    plot_displacement_error_histograms(error_displacements)
