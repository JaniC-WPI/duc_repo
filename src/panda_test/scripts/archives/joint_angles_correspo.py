#!/usr/bin/env python3
import json
import numpy as np
import cv2
import heapq
import os
import networkx as nx
import time
from sklearn.neighbors import KDTree, BallTree
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
import csv

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
    return float(distance)

def custom_distance(x, y):
    # Ensure x and y are in the format the model expects (flattened arrays)
    return predict_custom_distance(x, y, model)

def build_roadmap_with_kdtree(configurations, configuration_ids, joint_angles_dict, model, k_neighbors):
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

    tree1 = BallTree(flattened_configs, metric=custom_distance)

    G1 = nx.Graph()

    for i, (config, config_id) in enumerate(zip(configurations, configuration_ids)):
        G1.add_node(i, configuration=config, joint_angles=joint_angles_dict[config_id])

    for i, config in enumerate(flattened_configs):
        _, indices = tree1.query([config], k=k_neighbors + 1)  # +1 to include self in results

        for j in indices[0]:  # Skip self
            if j !=i:
                G1.add_edge(i, j)    

    tree2 = KDTree(flattened_configs)

    G2 = nx.Graph()

    for i, (config, config_id) in enumerate(zip(configurations, configuration_ids)):
        G2.add_node(i, configuration=config, joint_angles=joint_angles_dict[config_id])

    for i, config in enumerate(flattened_configs):
        _, indices = tree2.query([config], k=k_neighbors + 1)  # +1 to include self in results

        for j in indices[0]:  # Skip self
            if j !=i:
                G2.add_edge(i, j)    

    return G1, tree1, G2, tree2

# Function to add a configuration to the roadmap with collision checking
def add_config_to_roadmap(config, joint_angles, G, tree, k_neighbors, obstacle_center, half_diagonal, safe_distance):
    flattened_config = config.flatten().reshape(1, -1)
    _, indices = tree.query(flattened_config, k=k_neighbors)
    connections = 0    
    node_id = len(G.nodes)
    G.add_node(node_id, configuration=config, joint_angles=joint_angles)
    obstacle_boundary = geom.Polygon([
        (obstacle_center[0] - (half_diagonal + safe_distance), obstacle_center[1] - (half_diagonal + safe_distance)),
        (obstacle_center[0] + (half_diagonal + safe_distance), obstacle_center[1] - (half_diagonal + safe_distance)),
        (obstacle_center[0] + (half_diagonal + safe_distance), obstacle_center[1] + (half_diagonal + safe_distance)),
        (obstacle_center[0] - (half_diagonal + safe_distance), obstacle_center[1] + (half_diagonal + safe_distance)),
    ])
    
    for i in indices[0]:
        neighbor_config = G.nodes[i]['configuration']
        if is_collision_free(config, neighbor_config, obstacle_center, half_diagonal, safe_distance):
            # visualize_interactions(config, neighbor_config, obstacle_boundary)
            G.add_edge(node_id, i)

    if nx.is_connected(G):
        print("Roadmap is connected")
    else:
        print("Roadmap is disconnected")  
    
    return node_id

def is_collision_free(configuration1, configuration2, obstacle_center, half_diagonal, safe_distance):
    # Define the square boundary of the obstacle including the safe distance
    obstacle_boundary = geom.Polygon([
        (obstacle_center[0] - (half_diagonal + safe_distance), obstacle_center[1] - (half_diagonal + safe_distance)),
        (obstacle_center[0] + (half_diagonal + safe_distance), obstacle_center[1] - (half_diagonal + safe_distance)),
        (obstacle_center[0] + (half_diagonal + safe_distance), obstacle_center[1] + (half_diagonal + safe_distance)),
        (obstacle_center[0] - (half_diagonal + safe_distance), obstacle_center[1] + (half_diagonal + safe_distance)),
    ])

    # Check for collision between consecutive keypoints within the same configuration
    for config in [configuration1, configuration2]:
        for i in range(len(config) - 1):
            segment = geom.LineString([config[i], config[i+1]])
            if segment.intersects(obstacle_boundary):
                # print("collision detected")
                # If any segment intersects, the configuration is not collision-free
                return False
        
    for i in range(len(configuration1)):
        segment = geom.LineString([configuration1[i], configuration2[i]])
        if segment.intersects(obstacle_boundary):
            # print("edge collision detected")
            return False
        
     # If no segments intersect, the configuration is collision-free
    return True

def validate_and_remove_invalid_edges(G, obstacle_center, half_diagonal, safe_distance):
    # Iterate over a copy of the edges list to avoid modification issues during iteration
    for (u, v) in list(G.edges):
        config_u = G.nodes[u]['configuration']
        config_v = G.nodes[v]['configuration']
        # Perform the collision check for the edge
        if not is_collision_free(config_u, config_v, obstacle_center, half_diagonal, safe_distance):
            # If the edge is not collision-free, remove it from the graph
            G.remove_edge(u, v)
            # print(f"Removed invalid edge: {u} <-> {v}")

def find_path(G, start_node, goal_node):
    path_indices = nx.astar_path(G, source=start_node, target=goal_node)
    path_configurations = [(G.nodes[i]['configuration'], G.nodes[i]['joint_angles']) for i in path_indices]

    return path_configurations

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

if __name__ == "__main__":
    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged/'  # Replace with the path to your JSON files
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b128_e500_v32.pth'

    skip_step = 10
    start_index = 1
    end_index = 25000

    configurations, configuration_ids = load_keypoints_from_json(directory)
    joint_angles_dict = load_joint_angles_from_json(directory)
    model = load_model_for_inference(model_path)

    skipped_configs, skipped_ids = skip_configurations(configurations, configuration_ids, skip_step, start_index, end_index)

    num_neighbors = 25  # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    roadmap1, tree1, roadmap2, tree2 = build_roadmap_with_kdtree(skipped_configs, skipped_ids, joint_angles_dict, model, num_neighbors)
    end_time = time.time()

    print("Time taken to find the graph:", end_time - start_time)     

    # Define start and goal configurations as numpy arrays
    start_config = np.array([[250, 442], [252, 311], [215, 273], [172, 234], [192, 212], [220, 147], [249, 82], [248, 52], [286, 48]])
    goal_config = np.array([[250, 442], [252, 311], [275, 255], [294, 200], [322, 209], [394, 194], [468, 181], [494, 158], [522, 187]])

    start_joint_angles = np.array([0.9331, -1.33819, 2.2474])
    goal_joint_angles = np.array([0.267307, -1.38323, 2.58668])

    SAFE_ZONE = 40 
    obstacle_center = (400, 120)

    half_diagonal = 20
    # safe_distance = SAFE_ZONE

    obstacle_boundary = geom.Polygon([
        (obstacle_center[0] - (half_diagonal + SAFE_ZONE), obstacle_center[1] - (half_diagonal + SAFE_ZONE)),
        (obstacle_center[0] + (half_diagonal + SAFE_ZONE), obstacle_center[1] - (half_diagonal + SAFE_ZONE)),
        (obstacle_center[0] + (half_diagonal + SAFE_ZONE), obstacle_center[1] + (half_diagonal + SAFE_ZONE)),
        (obstacle_center[0] - (half_diagonal + SAFE_ZONE), obstacle_center[1] + (half_diagonal + SAFE_ZONE)), 
    ])

    # Add start configuration to roadmap
    start_node = add_config_to_roadmap(start_config, start_joint_angles, roadmap1, tree1, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE)
    goal_node = add_config_to_roadmap(goal_config, goal_joint_angles, roadmap1, tree1, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE)

    validate_and_remove_invalid_edges(roadmap1, obstacle_center, half_diagonal, SAFE_ZONE)
    validate_and_remove_invalid_edges(roadmap2, obstacle_center, half_diagonal, SAFE_ZONE)
        
    # Find and print the path from start to goal
    path_custom = find_path(roadmap1, start_node, goal_node)

    path_euclidean = find_path(roadmap2, start_node, goal_node)

    if path_custom:
         # Extract keypoints and joint angles from the path
        keypoint_path = [config[0] for config in path_custom]
        joint_angle_path = [config[1] for config in path_custom]

        print("Keypoint Path:", keypoint_path)
        print("Joint Angles Path:", joint_angle_path)
        
        # Save keypoints and joint angles to CSV
        save_keypoints_and_joint_angles_to_csv(path_custom, "/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_angles_correspondence/52/path_custom_kp_jt.csv")
    
        print("Data successfully saved to CSV.")

        point_set = []
        goal_sets = []
        # Iterate through the path, excluding the first and last configuration
        for configuration in path_custom[0:-1]:
            # Extract the last three keypoints of each configuration
            selected_points = configuration[[3, 4, 6, 7, 8]]
            selected_points_float = [[float(point[0]), float(point[1])] for point in selected_points]
            # Append these points to the point_set list
            point_set.append(selected_points_float)

        # Iterate through the path, excluding start and goal            
        for configuration in path_custom[1:]: 
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

        with open("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_angles_correspondence/52/cutom_dl_multi_features.yaml", "w") as yaml_file:
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
            for config in path_custom:
                file.write(str(config.tolist()) + "\n")
            file.write("\nPoint Set:\n")
            for points in point_set:
                file.write(str(points) + "\n")

        with open("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_angles_correspondence/52/path_configurations_custom.txt", "w") as file:
            file.write("Start Configuration:\n")
            file.write(str(start_config.tolist()) + "\n\n")
            file.write("Goal Configuration:\n")
            file.write(str(goal_config.tolist()) + "\n\n")
            file.write("Obstacle Parameters:\n")
            file.write("Path:\n")
            for config in path_custom:
                file.write(str(config.tolist()) + "\n")
            file.write("\nPoint Set:\n")
            for points in point_set:
                file.write(str(points) + "\n")

        print("Configurations successfully saved to configurations.txt")
    else:
        print("No path found.")

    if path_euclidean:
         # Extract keypoints and joint angles from the path
        keypoint_path = [config[0] for config in path_euclidean]
        joint_angle_path = [config[1] for config in path_euclidean]

        print("Keypoint Path:", keypoint_path)
        print("Joint Angles Path:", joint_angle_path)
        
        # Save keypoints and joint angles to CSV
        save_keypoints_and_joint_angles_to_csv(path_euclidean, "/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_angles_correspondence/52/euclidean_kp_jt.csv")
    
        print("Data successfully saved to CSV.")

        point_set = []
        goal_sets = []
        # Iterate through the path, excluding the first and last configuration
        for configuration in path_euclidean[0:-1]:
            # Extract the last three keypoints of each configuration
            selected_points = configuration[[3, 4, 6, 7, 8]]
            selected_points_float = [[float(point[0]), float(point[1])] for point in selected_points]
            # Append these points to the point_set list
            point_set.append(selected_points_float)

        # Iterate through the path, excluding start and goal            
        for configuration in path_euclidean[1:]: 
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

        with open("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_angles_correspondence/52/custom_dl_multi_features.yaml", "w") as yaml_file:
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
            for config in path_custom:
                file.write(str(config.tolist()) + "\n")
            file.write("\nPoint Set:\n")
            for points in point_set:
                file.write(str(points) + "\n")

        with open("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_angles_correspondence/52/path_configurations_euclidean.txt", "w") as file:
            file.write("Start Configuration:\n")
            file.write(str(start_config.tolist()) + "\n\n")
            file.write("Goal Configuration:\n")
            file.write(str(goal_config.tolist()) + "\n\n")
            file.write("Obstacle Parameters:\n")
            file.write("Path:\n")
            for config in path_custom:
                file.write(str(config.tolist()) + "\n")
            file.write("\nPoint Set:\n")
            for points in point_set:
                file.write(str(points) + "\n")

        print("Configurations successfully saved to configurations.txt")
    else:
        print("No path found.")


