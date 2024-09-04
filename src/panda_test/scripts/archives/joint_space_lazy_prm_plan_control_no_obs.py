#!/usr/bin/env python3
import json
import numpy as np
import cv2
import heapq
import os
import networkx as nx
import time
from sklearn.neighbors import KDTree
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
SAFE_DISTANCE = 30  # Safe distance from the obstacle

def load_keypoints_from_json(directory):
    configurations = []
    configuration_ids = []
    for filename in os.listdir(directory):
        if filename.endswith('.json') and not filename.endswith('_joint_angles.json') and not filename.endswith('_vel.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]  # Extracting x, y coordinates
                configurations.append(np.array(keypoints))
                configuration_ids.append(data['id'])  # Store the configuration ID
    return configurations, configuration_ids

def load_joint_angles_from_json(directory):
    joint_angles_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('_joint_angles.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                joint_angles_dict[data['id']] = np.array(data['joint_angles'])
    return joint_angles_dict

def skip_configurations(configurations, configuration_ids, skip_step=5, start=1, end=13000):
    skipped_configs = configurations[start:end:skip_step]
    skipped_ids = configuration_ids[start:end:skip_step]
    return skipped_configs, skipped_ids

def build_roadmap_with_kdtree(configurations, configuration_ids, joint_angles_dict, k_neighbors):
    """
    Build a roadmap using a KDTree for efficient nearest neighbor search based on joint angles.
    
    Args:
    - configurations: List[np.array], a list of configurations (keypoints flattened).
    - configuration_ids: List[int], a list of configuration IDs corresponding to each configuration.
    - joint_angles_dict: Dict, a dictionary mapping configuration ids to their joint angles.
    - k_neighbors: int, the number of neighbors to connect to each node.
    
    Returns:
    - G: nx.Graph, the constructed roadmap.
    - tree: KDTree, the KDTree built from joint angles.
    """
    # Flatten the configurations
    flattened_configs = np.vstack([config.flatten() for config in configurations])
    
    # Extract joint angles using the configuration IDs
    joint_angles_list = [joint_angles_dict[config_id] for config_id in configuration_ids]
    joint_angles_array = np.vstack(joint_angles_list)
    
    # Build KDTree using joint angles
    tree = KDTree(joint_angles_array)

    G = nx.Graph()

    for i, (config, config_id) in enumerate(zip(configurations, configuration_ids)):
        G.add_node(i, configuration=config)

    # Query KDTree for nearest neighbors
    for i, joint_angles in enumerate(joint_angles_array):
        distances, indices = tree.query([joint_angles], k=k_neighbors + 1)  # +1 to include the node itself
        
        for d, j in zip(distances[0], indices[0]):
            if i != j:  # Avoid self-loops
                joint_angles_i = joint_angles_array[i]
                joint_angles_j = joint_angles_array[j]
                
                # Calculate the joint displacement (distance) between two configurations
                joint_displacement = np.linalg.norm(joint_angles_i - joint_angles_j)
                distance = d

                # Only add edge if there is no collision (collision check outside this function)
                G.add_edge(i, j, weight=joint_displacement)  # Use joint displacement as edge weight
                 
                # Debugging statement to trace edge creation
                print(f"Edge added between Node {i} and Node {j} with joint displacement {joint_displacement} and joint distance {distance}")

    return G, tree

def add_config_to_roadmap(config, joint_angles, G, tree, k_neighbors):
    """
    Add a configuration with given joint angles to the roadmap using KDTree for nearest neighbor search.
    
    Args:
    - config: np.array, the configuration (keypoints) to add.
    - joint_angles: np.array, the joint angles corresponding to the configuration.
    - G: nx.Graph, the existing roadmap graph.
    - tree: KDTree, the KDTree built from joint angles.
    - k_neighbors: int, the number of neighbors to connect to the new node.
    
    Returns:
    - node_id: int, the ID of the newly added node in the roadmap.
    """
    flattened_config = config.flatten().reshape(1, -1)

    # Query KDTree for nearest neighbors based on joint angles
    distances, indices = tree.query([joint_angles], k=k_neighbors)

    node_id = len(G.nodes)
    G.add_node(node_id, configuration=config)

    for idx, dist in zip(indices[0], distances[0]):
        G.add_edge(node_id, idx, weight=dist)

    return node_id

def find_path(G, start_node, goal_node):
    try:
        path_indices = nx.astar_path(G, source=start_node, target=goal_node)
        path_configurations = [G.nodes[i]['configuration'] for i in path_indices]
        return path_configurations
    except nx.NetworkXNoPath:
        print("No path found.")
        return None

# Main execution
if __name__ == "__main__":
    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged/'  # Replace with the path to your JSON files
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b128_e500_v34.pth'
    num_samples = 500

    skip_step = 10
    start_index = 1
    end_index = 25000

    configurations, configuration_ids = load_keypoints_from_json(directory)
    joint_angles_dict = load_joint_angles_from_json(directory)

    skipped_configs, skipped_ids = skip_configurations(configurations, configuration_ids, skip_step, start_index, end_index)

    # Parameters for PRM
    num_neighbors = 25  # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    roadmap, tree = build_roadmap_with_kdtree(skipped_configs, skipped_ids, joint_angles_dict, num_neighbors)
    end_time = time.time()

    print("Time taken to find the graph:", end_time - start_time)      

    # Define start and goal configurations as numpy arrays
    start_config = np.array([[250, 442], [252, 311], [215, 273], [172, 234], [192, 212], [220, 147], [249, 82], [248, 52], [286, 48]])
    goal_config = np.array([[250, 442], [252, 311], [275, 255], [294, 200], [322, 209], [394, 194], [468, 181], [494, 158], [522, 187]])

    start_joint_angles = np.array([0.9331, -1.33819, 2.2474])
    goal_joint_angles = np.array([0.267307, -1.38323, 2.58668])

    # Add start configuration to roadmap
    start_node = add_config_to_roadmap(start_config, start_joint_angles, roadmap, tree, num_neighbors)
    goal_node = add_config_to_roadmap(goal_config, goal_joint_angles, roadmap, tree, num_neighbors)
        
    # Find and print the path from start to goal
    path = find_path(roadmap, start_node, goal_node)

    if path:
        point_set = []
        goal_sets = []
        # Iterate through the path, excluding the first and last configuration
        for configuration in path[0:-1]:
            # Extract the last three keypoints of each configuration
            selected_points = configuration[[3, 4, 6, 7, 8]]
            selected_points_float = [[float(point[0]), float(point[1])] for point in selected_points]
            # Append these points to the point_set list
            point_set.append(selected_points_float)

        # Iterate through the path, excluding start and goal            
        for configuration in path[1:]: 
            selected_points = configuration[[3, 4, 6, 7, 8]]
            selected_points_float = [[float(point[0]), float(point[1])] for point in selected_points]
            goal_features = []
            for point in selected_points_float:
                goal_features.extend(point)  # Add x, y as a pair
            goal_sets.append(goal_features)

        print("Point Set:", point_set)
        print("Goal Sets:", goal_sets)
    
        with open("/home/jc-merlab/duc_repo/src/panda_test/config/dl_multi_features.yaml", "w") as yaml_file:
            s = "dl_controller:\n"
            s += "  num_goal_sets: " + str(len(goal_sets)) + "\n"
            for i, goal in enumerate(goal_sets, start=1):
                # Convert the list of floats into a comma-separated string
                goal_str = ', '.join(map(str, goal))
                s += f"  goal_features{i}: [{goal_str}]\n"
    
            # Write the string to the file
            yaml_file.write(s)

        with open("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space/41/dl_multi_features.yaml", "w") as yaml_file:
            s = "dl_controller:\n"
            s += "  num_goal_sets: " + str(len(goal_sets)) + "\n"
            for i, goal in enumerate(goal_sets, start=1):
                # Convert the list of floats into a comma-separated string
                goal_str = ', '.join(map(str, goal))
                s += f"  goal_features{i}: [{goal_str}]\n"
    
            # Write the string to the file
            yaml_file.write(s)
    
        print("Data successfully written to config/dl_multi_features.yaml")

        # Save configurations to a .txt file
        with open("/home/jc-merlab/duc_repo/src/panda_test/config/path_configurations.txt", "w") as file:
            file.write("Start Configuration:\n")
            file.write(str(start_config.tolist()) + "\n\n")
            file.write("Goal Configuration:\n")
            file.write(str(goal_config.tolist()) + "\n\n")
            file.write("Obstacle Parameters:\n")
            file.write("Path:\n")
            for config in path:
                file.write(str(config.tolist()) + "\n")
            file.write("\nPoint Set:\n")
            for points in point_set:
                file.write(str(points) + "\n")

        with open("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space/41/path_configurations.txt", "w") as file:
            file.write("Start Configuration:\n")
            file.write(str(start_config.tolist()) + "\n\n")
            file.write("Goal Configuration:\n")
            file.write(str(goal_config.tolist()) + "\n\n")
            file.write("Obstacle Parameters:\n")
            file.write("Path:\n")
            for config in path:
                file.write(str(config.tolist()) + "\n")
            file.write("\nPoint Set:\n")
            for points in point_set:
                file.write(str(points) + "\n")

        print("Configurations successfully saved to configurations.txt")
    else:
        print("No path found.")
