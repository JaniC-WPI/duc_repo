#!/usr/bin/env python3
import json
import numpy as np
import cv2
import heapq
import random
import os
import networkx as nx
import time
from sklearn.neighbors import KDTree, BallTree
# from scipy.spatial import KDTree 
import torchvision
from PIL import Image
import torch
import yaml
import shapely.geometry as geom
import scipy
import matplotlib.pyplot as plt
from pos_regression_control import PosRegModel
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import pickle, csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_model_for_inference(model_path):    
    model = PosRegModel(18)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to inference mode
    return model

def predict_custom_distance(current_config, next_config, model):
    # Convert to 2D tensors if necessary
    start_kp_flat = torch.tensor(current_config.flatten(), dtype=torch.float).unsqueeze(0)  # Add batch dimension
    next_kp_flat = torch.tensor(next_config.flatten(), dtype=torch.float).unsqueeze(0)  # Add batch dimension

    # Predict the next configuration
    with torch.no_grad():
        output = model(start_kp_flat, next_kp_flat).squeeze(0).numpy()  # Remove batch dimension for output

    distance = np.linalg.norm(output)
    return float(distance)  # Reshape to the original configuration format

def custom_distance(x, y):
    # Ensure x and y are in the format the model expects (flattened arrays)
    return predict_custom_distance(x, y, model)

# Load the roadmap and KDTree from files
def load_graph_and_tree(graph_path, tree_path):
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    with open(tree_path, 'rb') as f:
        tree = pickle.load(f)
    print(f"Graph loaded from {graph_path}")
    print(f"KDTree loaded from {tree_path}")
    return graph, tree

def add_config_to_roadmap_no_obs(config, joint_angles, G, tree, k_neighbors):
    """Add a configuration to the roadmap, connecting it to its k nearest neighbors."""
    flattened_config = config.flatten().reshape(1, -1)
    dist, indices = tree.query(flattened_config, k=k_neighbors)
    
    node_id = len(G.nodes)
    G.add_node(node_id, configuration=config, joint_angles=joint_angles)
    
    for d,i in zip(dist[0],indices[0]):
        G.add_edge(node_id, i, weight=d)
    
    return node_id

def find_path(G, start_node, goal_node):
    # path_indices = nx.dijkstra_path(G, source=start_node, target=goal_node, weight='weight')
    path_indices = nx.astar_path(G, source=start_node, target=goal_node)
    path_configurations = [[G.nodes[i]['configuration'], G.nodes[i]['joint_angles']] for i in path_indices]

    return path_configurations

def astar_custom(graph, start, goal, heuristic_func):
    # Priority queue (min-heap) to hold nodes to be evaluated
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    # Dictionaries to hold the cost of the shortest path to a node and the path to reach it
    g_costs = {start: 0}
    came_from = {start: None}
    
    # While there are nodes to evaluate
    while open_set:
        # Get the node with the lowest f(n) = g(n) + h(n) value
        _, current = heapq.heappop(open_set)

        # If we reached the goal, reconstruct the path
        if current == goal:
            return reconstruct_path(came_from, start, goal)

        # Explore neighbors using networkx graph
        for neighbor in graph.neighbors(current):
            # Access the edge weight between current and neighbor
            weight = graph.edges[current, neighbor]['weight']
            # Calculate tentative g cost
            tentative_g_cost = g_costs[current] + weight

            # If this path to neighbor is better, update the costs and the path
            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic_func(neighbor, goal)
                heapq.heappush(open_set, (f_cost, neighbor))
                came_from[neighbor] = current

    # If the goal was not reached
    return None


def reconstruct_path(came_from, start, goal):
    path = []
    current = goal
    while current:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def edge_weight_heuristic(graph, current_node, goal_node):
    # If there is a direct edge, return its weight
    if graph.has_edge(current_node, goal_node):
        return graph.edges[current_node, goal_node]['weight']
    return 0

# def find_path(G, start_node, goal_node):
#     path_indices = astar_custom(G, start_node, goal_node, lambda u, v: edge_weight_heuristic(G, u, v))
    
#     path_configurations = [[G.nodes[i]['configuration'], G.nodes[i]['joint_angles']] for i in path_indices]
        
#     return path_configurations

def save_keypoints_and_joint_angles_to_csv(path, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Define headers
        headers = ['Config'] + [f'KP_{i}_x' for i in range(len(path[0][0]))] + [f'KP_{i}_y' for i in range(len(path[0][0]))] + ['Joint 1', 'Joint 2', 'Joint 3']
        csv_writer.writerow(headers)

        # Write each configuration and its joint angles
        for i, (config, angles) in enumerate(path):
            flat_config = [coord for kp in config for coord in kp]  # Flatten the keypoint configuration
            row = [f'Config_{i}'] + flat_config + list(angles)
            csv_writer.writerow(row)

def save_path_with_distances_to_csv(path, filename, model):
    """
    Saves the path, distances between configurations, and joint angle distances to a CSV file.
    
    Args:
    - path: List of configurations and joint angles.
    - filename: Name of the CSV file to save the data.
    - model: The model used for calculating custom distances between keypoints.
    """
    kp_distances = []
    joint_angle_distances = []

    # Calculate distances between consecutive configurations and joint angles
    for i in range(1, len(path)):
        current_config = path[i-1][0]
        next_config = path[i][0]
        current_angles = path[i-1][1]
        next_angles = path[i][1]

        # Distance between keypoint configurations
        kp_distance = predict_custom_distance(current_config, next_config, model)
        kp_distances.append(kp_distance)

        # Euclidean distance between joint angles
        joint_angle_distance = np.linalg.norm(np.array(next_angles) - np.array(current_angles))
        joint_angle_distances.append(joint_angle_distance)

    # Write the configurations, joint angles, keypoint distances, and joint angle distances to a CSV file
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Define headers
        headers = ['Config'] + [f'KP_{i}_x' for i in range(len(path[0][0]))] + \
                  [f'KP_{i}_y' for i in range(len(path[0][0]))] + \
                  ['Joint 1', 'Joint 2', 'Joint 3', 'Distance to next KP', 'Distance to next Joint Angles']
        csv_writer.writerow(headers)

        # Write each configuration and its joint angles
        for i, (config, angles) in enumerate(path):
            flat_config = [coord for kp in config for coord in kp]  # Flatten the keypoint configuration
            if i < len(kp_distances):
                row = [f'Config_{i}'] + flat_config + list(angles) + [kp_distances[i], joint_angle_distances[i]]
            else:
                row = [f'Config_{i}'] + flat_config + list(angles) + ['', '']  # No distance for the last configuration
            csv_writer.writerow(row)

    print(f"Path, keypoint distances, and joint angle distances successfully saved to {filename}")

def save_image_with_points(image_path, output_path, output_dir, points_sets):
    """
    Draws points and lines on an image and saves it to specified directories.

    Args:
    - image_path (str): Path to the input image.
    - output_path (str): Path to save the final modified image.
    - output_dir (str): Directory to save intermediate images.
    - points_sets (list): List of sets of points to draw.

    Returns:
    - None
    """
    
    # Load your image
    static_image = cv2.imread(image_path)
    
    if static_image is None:
        print(f"Error: Could not read the image at {image_path}.")
        return
    
    # Make a copy of the image for drawing
    gif_image = static_image.copy()

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define fixed colors for the points
    fixed_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    colors = np.random.randint(0, 255, (len(points_sets), 3))

    # Draw points and lines on the image
    for set_index, (points, color) in enumerate(zip(points_sets, colors)):
        if set_index == 0:
            for x, y in points:
                cv2.circle(static_image, (int(x), int(y)), 9, (0, 255, 0), -1)
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                cv2.line(static_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=4)
        else:
            for index, (x, y) in enumerate(points):
                cv2.circle(static_image, (int(x), int(y)), 9, fixed_colors[index], -1)
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                cv2.line(static_image, (int(x1), int(y1)), (int(x2), int(y2)), tuple(int(c) for c in color), thickness=4)

        # Save the intermediate image with path number
        cv2.imwrite(os.path.join(output_dir, f'path_{set_index}.jpg'), static_image)

    # Save the final modified image
    cv2.imwrite(output_path, static_image)

def create_goal_image(goal_config, output_path):
    """
    Creates an image with specified keypoints and lines connecting them.

    Args:
    - goal_config (np.ndarray): Array of keypoints (x, y coordinates).
    - image_size (tuple): Size of the output image (height, width, channels).
    - circle_radius (int): Radius of the circles to draw.
    - circle_color (tuple): Color of the circles (BGR format).
    - line_color (tuple): Color of the lines (BGR format).
    - line_thickness (int): Thickness of the lines.
    - output_path (str): Path to save the output image.

    Returns:
    - None
    """
    # Initialize the blank image
    goal_image = np.zeros((480,640,3), dtype=np.int8)

    # Draw circles at each point in goal_config
    for point in goal_config:
        cv2.circle(goal_image, tuple(point.astype(int)), radius=9, color=(0, 0, 255), thickness=-1)

    # Draw lines between consecutive points in goal_config
    for i in range(len(goal_config) - 1):
        cv2.line(goal_image, tuple(goal_config[i].astype(int)), tuple(goal_config[i+1].astype(int)), (0, 0, 255), 4)

    # Save the image to the specified path
    cv2.imwrite(output_path, goal_image)
    print(f"Goal image successfully saved to {output_path}")

    # Function to plot joint angles in 3D
def plot_joint_angles_3d(path, output_path):
    joint_angles = [config[1] for config in path]  # Extract joint angles from the path

    joint_1 = [angles[0] for angles in joint_angles]
    joint_2 = [angles[1] for angles in joint_angles]
    joint_3 = [angles[2] for angles in joint_angles]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(joint_1, joint_2, joint_3, marker='o')

    ax.set_xlabel('Joint 1')
    ax.set_ylabel('Joint 2')
    ax.set_zlabel('Joint 3')

    plt.title('3D Joint Angle Plot')
    plt.savefig(output_path)
    plt.close()

# Function to calculate joint angle distance from start to each configuration
def calculate_joint_angle_distances(path):
    joint_angle_distances = []
    start_joint_angles = np.array(path[0][1])  # The joint angles of the start configuration

    for config in path:
        current_joint_angles = np.array(config[1])
        distance = np.linalg.norm(current_joint_angles - start_joint_angles)
        joint_angle_distances.append(distance)

    return joint_angle_distances

# Function to randomly pick start and goal configurations from the roadmap
# Function to randomly pick start and goal configurations from the roadmap
def random_start_goal_from_roadmap(G, num_trials):
    nodes = list(G.nodes)
    start_goal_pairs = []
    
    for _ in range(num_trials):
        start_node = random.choice(nodes)
        goal_node = random.choice(nodes)
        
        # Ensure start and goal nodes are different
        while start_node == goal_node:
            goal_node = random.choice(nodes)
        
        start_goal_pairs.append((start_node, goal_node))
    
    return start_goal_pairs

# Function to plot joint angles in 3D for all three roadmaps in the same figure
def plot_joint_angles_3d_all_roadmaps(paths, labels, output_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Mark the start and goal joint angles    

    # Plot joint angles for each roadmap
    for path, label in zip(paths, labels):
        joint_angles = [config[1] for config in path]
        joint_1 = [angles[0] for angles in joint_angles]
        joint_2 = [angles[1] for angles in joint_angles]
        joint_3 = [angles[2] for angles in joint_angles]
        ax.plot(joint_1, joint_2, joint_3, marker='o', label=label)
        if label == 'Euclidean':
            ax.scatter(joint_1[0], joint_2[0], joint_3[0], color='purple', marker='o', s=100, label='Start')
            ax.scatter(joint_1[-1], joint_2[-1], joint_3[-1], color='purple', marker='X', s=100, label='Goal')

    ax.set_xlabel('Joint 1')
    ax.set_ylabel('Joint 2')
    ax.set_zlabel('Joint 3')
    ax.legend()

    plt.title('3D Joint Angle Plot for All Roadmaps')
    plt.savefig(output_path)
    plt.close()

# Function to calculate total joint angle distances for a roadmap
def calculate_total_joint_angle_distance(path):
    total_distance = 0

    # Iterate over pairs of consecutive configurations
    for i in range(1, len(path)):
        previous_joint_angles = np.array(path[i - 1][1])
        current_joint_angles = np.array(path[i][1])
        # Calculate the Euclidean distance between the consecutive joint angles
        distance = np.linalg.norm(current_joint_angles - previous_joint_angles)
        total_distance += distance

    return total_distance

import pandas as pd
import matplotlib.pyplot as plt

def plot_total_joint_distances(csv_file, output_image_path):
    """
    Plots total joint angle distances for all trials and compares the distances between different methods.
    
    Args:
    - csv_file (str): Path to the CSV file containing the total joint distances for each trial.
    - output_image_path (str): Path to save the output comparison plot image.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Extract trial numbers and distances for each method
    trials = df['Trial']
    ground_truth_distances = df['Ground Truth']
    custom_distances = df['Custom']
    euclidean_distances = df['Euclidean']

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot distances for each method
    plt.plot(trials, ground_truth_distances, label='Ground Truth', color='green', marker='o')
    plt.plot(trials, custom_distances, label='Custom', color='blue', marker='o')
    plt.plot(trials, euclidean_distances, label='Euclidean', color='red', marker='o')

    # Set plot labels and title
    plt.xlabel('Trial')
    plt.ylabel('Total Joint Angle Distance')
    plt.title('Total Joint Angle Distances Comparison for 100 Trials')
    plt.legend()

    # Save the plot as an image
    plt.savefig(output_image_path)
    plt.show()

def calculate_jerk(joint_positions):
    """
    Calculates the jerk (rate of change of acceleration) for a given sequence of joint positions.
    
    Args:
    - joint_positions (pd.Series): A pandas series of joint positions (e.g., Joint 1, Joint 2, Joint 3).
    
    Returns:
    - jerk: A measure of the smoothness of the trajectory.
    """
    # Calculate first and second derivatives (acceleration and jerk)
    velocity = np.gradient(joint_positions)  # First derivative
    acceleration = np.gradient(velocity)     # Second derivative (acceleration)
    jerk = np.gradient(acceleration)         # Third derivative (jerk)
    
    # Return the norm of the jerk as a measure of smoothness
    return np.linalg.norm(jerk)

def calculate_total_variation(joint_positions):
    """
    Calculates the total variation for a given sequence of joint positions.
    The smaller the variation, the smoother the trajectory.

    Args:
    - joint_positions (pd.Series): A pandas series of joint positions (e.g., Joint 1, Joint 2, Joint 3).
    
    Returns:
    - total_variation: The sum of absolute differences between consecutive joint positions.
    """
    variation = np.abs(np.diff(joint_positions))
    total_variation = np.sum(variation)
    return total_variation

# Example usage

def automate_path_creation_for_all_distances(roadmaps, trees, num_neighbors, model_path, output_folders, num_trials=100):
    roadmap_labels = ["Ground Truth", "Custom", "Euclidean"]  # Labels for each roadmap

    # Initialize list to store total distances for each trial
    total_joint_distances = []
    jerk_data = []
    variation_data = []

    trial_count = 0  # Track the number of completed trials
    while trial_count < num_trials:
        valid_start_goal = False
        
        # Generate a new random start/goal pair until valid for all roadmaps
        while not valid_start_goal:
            start_node, goal_node = random_start_goal_from_roadmap(roadmaps[0], 1)[0]
            valid_start_goal = all(start_node in roadmap.nodes and goal_node in roadmap.nodes for roadmap in roadmaps)
        
        trial_joint_distances = [trial_count]  # First column in CSV will be trial number
        jerk_trial = [trial_count]  # First column in CSV will be trial number
        variation_trial = [trial_count]  # First column in CSV will be trial number
        
        paths = []  # Store paths for each roadmap for plotting
        for roadmap_idx in range(len(roadmaps)):
            roadmap = roadmaps[roadmap_idx]
            tree = trees[roadmap_idx]
            label = roadmap_labels[roadmap_idx]
            folder_no_obs = output_folders[roadmap_idx]

            print(f"Processing roadmap: {label} for trial {trial_count}")

            # Load model for this roadmap
            model = load_model_for_inference(model_path)

            # Get start and goal configurations from the nodes in the roadmap
            start_config = roadmap.nodes[start_node]['configuration']
            goal_config = roadmap.nodes[goal_node]['configuration']
            start_joint_angles = roadmap.nodes[start_node]['joint_angles']
            goal_joint_angles = roadmap.nodes[goal_node]['joint_angles']
            
            # Create folders for this trial
            trial_folder_no_obs = os.path.join(folder_no_obs, f'trial_{trial_count}')
            os.makedirs(trial_folder_no_obs, exist_ok=True)
            
            # Find the path without obstacles
            path_no_obs = find_path(roadmap, start_node, goal_node)
            paths.append(path_no_obs)  # Append path for plotting

            # Save the path without obstacles
            save_keypoints_and_joint_angles_to_csv(path_no_obs, os.path.join(trial_folder_no_obs, 'joint_keypoints.csv'))
            save_path_with_distances_to_csv(path_no_obs, os.path.join(trial_folder_no_obs, 'save_distances.csv'), model)

            if path_no_obs:
                # Calculate total joint angle distance and add to list for this trial
                total_distance = calculate_total_joint_angle_distance(path_no_obs)
                trial_joint_distances.append(total_distance)

                joint_angles = [config[1] for config in path_no_obs]
                
                # Calculate jerk and joint variation for each joint
                for joint_idx in range(3):
                    joint_positions = [angles[joint_idx] for angles in joint_angles]
                    jerk_value = calculate_jerk(joint_positions)
                    variation_value = calculate_total_variation(joint_positions)
                    
                    jerk_trial.append(jerk_value)
                    variation_trial.append(variation_value)


        # Plot joint angles for all three roadmaps in one figure
        plot_joint_angles_3d_all_roadmaps(paths, roadmap_labels, os.path.join(output_folders[0], f'joint_angles_3d_trial_{trial_count}.png'))
        total_joint_distances.append(trial_joint_distances)
        jerk_data.append(jerk_trial)
        variation_data.append(variation_trial)

        trial_count += 1

    # Save total joint angle distances for all trials to CSV
    with open(os.path.join(output_folders[0], 'total_joint_angle_distances_fresh_432_all.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['Trial', 'Ground Truth', 'Custom', 'Euclidean'])
        # Write data for each trial
        csv_writer.writerows(total_joint_distances)

    with open(os.path.join(output_folders[0], 'joint_variation_data.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        headers = ['Trial'] + [f'{label} Joint {i+1}' for label in roadmap_labels for i in range(3)]
        csv_writer.writerow(headers)
        # Write data for each trial
        csv_writer.writerows(variation_data)

    with open(os.path.join(output_folders[0], 'jerk_data.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        headers = ['Trial'] + [f'{label} Joint {i+1}' for label in roadmap_labels for i in range(3)]
        csv_writer.writerow(headers)
        # Write data for each trial
        csv_writer.writerows(jerk_data)    

def plot_boxplot_comparison(csv_file, output_image_path):
    """
    Creates a box plot to compare the total joint distances for each method.
    
    Args:
    - csv_file (str): Path to the CSV file containing the total joint distances for each trial.
    - output_image_path (str): Path to save the output comparison plot image.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Create a box plot for Ground Truth, Custom, and Euclidean distances
    plt.figure(figsize=(8, 6))
    df[['Ground Truth', 'Custom', 'Euclidean']].boxplot()

    # Set plot labels and title
    plt.ylabel('Total Joint Angle Distance')
    plt.title('Box Plot Comparison of Total Joint Distances')

    # Save the plot as an image
    plt.savefig(output_image_path)
    plt.show()

def plot_bar_comparison(csv_file, output_image_path):
    """
    Creates a bar plot showing the count of trials where each method has the largest total distance.
    
    Args:
    - csv_file (str): Path to the CSV file containing the total joint distances for each trial.
    - output_image_path (str): Path to save the output bar plot image.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Count how many times each method has the largest total distance
    largest_counts = {'Ground Truth': 0, 'Custom': 0, 'Euclidean': 0}
    for _, row in df.iterrows():
        max_method = row[['Ground Truth', 'Custom', 'Euclidean']].idxmax()
        largest_counts[max_method] += 1

    # Create a bar plot
    methods = list(largest_counts.keys())
    counts = list(largest_counts.values())

    plt.figure(figsize=(8, 6))
    plt.bar(methods, counts, color=['green', 'blue', 'red'])

    # Set plot labels and title
    # plt.ylabel('Count of Trials with Largest Distance')
    # plt.title('Comparison of Methods by Count of Trials with Largest Distance')

    # Save the plot as an image
    plt.savefig(output_image_path)
    plt.show()

def plot_histogram_comparison(csv_file, output_image_path):
    """
    Creates a histogram to compare the distribution of total joint distances for each method.
    
    Args:
    - csv_file (str): Path to the CSV file containing the total joint distances for each trial.
    - output_image_path (str): Path to save the output comparison plot image.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Create histograms for each method
    plt.figure(figsize=(10, 6))
    plt.hist(df['Ground Truth'], bins=20, alpha=0.5, label='Ground Truth', color='green')
    plt.hist(df['Custom'], bins=20, alpha=0.5, label='Custom', color='blue')
    plt.hist(df['Euclidean'], bins=20, alpha=0.5, label='Euclidean', color='red')

    # Set plot labels and title
    plt.xlabel('Total Joint Angle Distance')
    plt.ylabel('Frequency')
    plt.title('Histogram Comparison of Total Joint Distances')
    plt.legend()

    # Save the plot as an image
    plt.savefig(output_image_path)
    plt.show()



# Main function to execute
if __name__ == "__main__":
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged/'  # Replace with the path to your JSON files
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b128_e500_v33.pth'
    model = load_model_for_inference(model_path)
    # graph_paths = ['/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_roadmap_angle_fresh.pkl',\
    #                '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_roadmap_angle_fresh.pkl', \
    #               '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_roadmap_angle_fresh.pkl']
    # tree_paths = ['/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_tree_angle_fresh.pkl',\
    #               '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_tree_angle_fresh.pkl', \
    #              '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_tree_angle_fresh.pkl']
    # file_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/'

    # custom_no_obs = os.path.join(file_path, 'custom_fresh_trial')
    # euclidean_no_obs = os.path.join(file_path, 'euclidean_fresh_trial')
    # gt_no_obs = os.path.join(file_path, 'gt_fresh_trial')
    # output_folders = [gt_no_obs, custom_no_obs, euclidean_no_obs]

    # # Automate path creation for 100 different start and goal configurations
    # # Load the roadmaps and trees for all three distance types
    # roadmaps = [load_graph_and_tree(graph_paths[i], tree_paths[i])[0] for i in range(3)]
    # trees = [load_graph_and_tree(graph_paths[i], tree_paths[i])[1] for i in range(3)]

    if __name__ == "__main__":
    # Define the mappings between roadmaps, trees, and labels
        roadmap_mappings = [
            {
                "label": "Ground Truth",
                "graph_path": '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_roadmap_angle_fresh_432_all.pkl',
                "tree_path": '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_tree_angle_fresh_432_all.pkl',
                "output_folder": '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/gt_fresh_trial'
            },
            {
                "label": "Custom",
                "graph_path": '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_roadmap_angle_fresh_432_all.pkl',
                "tree_path": '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_tree_angle_fresh_432_all.pkl',
                "output_folder": '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_fresh_trial'
            },
            {
                "label": "Euclidean",
                "graph_path": '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_roadmap_angle_fresh_432_all.pkl',
                "tree_path": '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_tree_angle_fresh_432_all.pkl',
                "output_folder": '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_fresh_trial'
            }
        ]

        # Load the roadmaps and trees
        for mapping in roadmap_mappings:
            roadmap, tree = load_graph_and_tree(mapping['graph_path'], mapping['tree_path'])
            mapping['roadmap'] = roadmap
            mapping['tree'] = tree

        # Automate path creation for 100 different start and goal configurations
        num_trials = 100
        model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b128_e400_v32.pth'
        automate_path_creation_for_all_distances(
            [mapping['roadmap'] for mapping in roadmap_mappings],  # Roadmaps
            [mapping['tree'] for mapping in roadmap_mappings],      # Trees
            num_neighbors=25, 
            model_path=model_path,
            output_folders=[mapping['output_folder'] for mapping in roadmap_mappings],  # Output folders
            num_trials=num_trials
        )

        # Automate path creation for all three roadmaps and 100 start/goal pairs
        # automate_path_creation_for_all_distances(roadmaps, trees, num_neighbors=25, model_path=model_path,
        #                                          output_folders=output_folders, num_trials=100)


        # csv_file = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/gt_fresh_trial/total_joint_angle_distances_fresh.csv'
        # output_image_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_angle_comparison_plot.png'

        # plot_total_joint_distances(csv_file, output_image_path)

        # Example usage
        csv_file = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/gt_fresh_trial/total_joint_angle_distances_fresh_432_all.csv'
        output_image_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/bar_plot_comparison_plot.png'

        plot_bar_comparison(csv_file, output_image_path)

        # output_image_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/histogram_comparison_plot.png'

        # plot_histogram_comparison(csv_file, output_image_path)

