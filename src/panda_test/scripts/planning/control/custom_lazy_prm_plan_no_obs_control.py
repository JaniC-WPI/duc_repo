#!/usr/bin/env python3
import json
import numpy as np
import cv2
import heapq
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

# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Load keypoints from JSON files in a given directory
def load_keypoints_from_json(directory):
    configurations = []
    for filename in os.listdir(directory):
        # if filename.endswith('.json'):
        if filename.endswith('.json') and not filename.endswith('_joint_angles.json') and not filename.endswith('_vel.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                # Convert keypoints to integers
                keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]  # Extracting x, y coordinates
                configurations.append(np.array(keypoints))
    # print(configurations)
    # print("Shape of a single configuration:", configurations[0].shape)  
    return configurations

def load_keypoints_from_truncated_json(directory):
    configurations = []
    for filename in os.listdir(directory):
        # if filename.endswith('.json'):
        if filename.endswith('.json') and not filename.endswith('_joint_angles.json') and not filename.endswith('_vel.json'):
            file_index = int(filename.split('.')[0])
            if file_index >= 10000:
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    # Convert keypoints to integers
                    keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]  # Extracting x, y coordinates
                    configurations.append(np.array(keypoints))

    return configurations

def load_and_sample_configurations(directory, num_samples):
    # Load configurations from JSON files
    configurations = load_keypoints_from_json(directory)

    # If there are more configurations than needed, sample a subset
    if len(configurations) > num_samples:
        sampled_indices = np.random.choice(len(configurations), size=num_samples, replace=False)
        sampled_configurations = [configurations[i] for i in sampled_indices]
    else:
        sampled_configurations = configurations

    return sampled_configurations

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

def build_lazy_roadmap_with_kdtree(configurations, k_neighbors, model):
    """
    Build a LazyPRM roadmap using a KDTree for efficient nearest neighbor search.
    
    Args:
    - configurations: List[np.array], a list of configurations (keypoints flattened).
    - k_neighbors: int, the number of neighbors to connect to each node.
    
    Returns:
    - G: nx.Graph, the constructed roadmap.
    """
    configurations = configurations[1:15000:5]
    # print("Shape of configurations before building the roadmap:", len(configurations), configurations[0].shape)

    flattened_configs = np.vstack([config.flatten() for config in configurations])
    tree = BallTree(flattened_configs, metric=lambda x, y: predict_custom_distance(x, y, model))
    print("tree is built")
    # tree = KDTree(distance_matrix) 

    G = nx.Graph()
   # flattened_configs = flattened_configs[1:9000:10]
   # configurations = configurations[1:9000:10]

    for i, config in enumerate(configurations):
        G.add_node(i, configuration=config)

    for i, config in enumerate(flattened_configs):
        _, indices = tree.query([config], k=k_neighbors + 1)  # +1 to include self in results
        #indices = tree.query_radius(config.reshape(1,-1), r=20,count_only=False) # +1 to include self in results

        for j in indices[0]:  # Skip self
            if j !=i:
                G.add_edge(i, j)
        
    # print(G.nodes.items())
    #SG= G.subgraph(range(1,9000,100))
    pos_dict = {n[0]:n[1]["configuration"][5] for n in G.nodes.items()}      
    # print(pos_dict) 
    nx.draw_networkx(G,node_size=5,width=0.3, with_labels=False, pos=pos_dict)
    plt.show()        
    return G, tree

# Add a new configuration to the roadmap
def add_config_to_roadmap(config, G, tree, k_neighbors):
    """Add a configuration to the roadmap, connecting it to its k nearest neighbors."""
    flattened_config = config.flatten().reshape(1, -1)
    _, indices = tree.query(flattened_config, k=k_neighbors)
    
    node_id = len(G.nodes)
    G.add_node(node_id, configuration=config)
    
    for i in indices[0]:
        G.add_edge(node_id, i)
    
    # Update the tree with the new node for future queries
    new_flattened_configs = np.vstack([tree.data, flattened_config])
    new_tree = KDTree(new_flattened_configs)
    
    return node_id, new_tree

def distance_line_to_point(p1, p2, point):
    """Calculates the distance of a point to a line segment."""
    numerator = np.abs((p2[1] - p1[1]) * point[0] - (p2[0] - p1[0]) * point[1] + p2[0] * p1[1] - p2[1] * p1[0])
    denominator = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    return numerator / denominator

def find_path_heuristic(G, start_node, goal_node, heuristic):
    try:
        path_indices = nx.astar_path(G, source=start_node, target=goal_node, heuristic=heuristic, weight='cost')
        path_configurations = [G.nodes[i]['configuration'] for i in path_indices]
        return path_configurations
    except nx.NetworkXNoPath:
        print("No path found between the specified nodes.")
        return []
    
def find_path(G, start_node, goal_node):
    path_indices = nx.astar_path(G, source=start_node, target=goal_node)
    path_configurations = [G.nodes[i]['configuration'] for i in path_indices]

    return path_configurations

def plot_path_on_image_dir(image_path, path, start_config, goal_config, output_directory):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Base image
    base_image = cv2.imread(image_path)

    # Function to generate a color
    def path_color():
        return (255, 0, 0)  # Blue color for path

    # Draw start and goal keypoints
    for point in start_config:
        cv2.circle(base_image, tuple(point.astype(int)), radius=5, color=(0, 0, 255), thickness=-1)  # Red for start
    for point in goal_config:
        cv2.circle(base_image, tuple(point.astype(int)), radius=5, color=(0, 255, 0), thickness=-1)  # Green for goal

    # Draw and save each path configuration
    for idx, config_tuples in enumerate(path):
        image = base_image.copy()  # Copy the base image
        config = np.array(config_tuples)  # Convert config from tuple of tuples to numpy array
        for i in range(len(config) - 1):
            cv2.line(image, tuple(config[i].astype(int)), tuple(config[i+1].astype(int)), path_color(), 2)
        for point in config:
            cv2.circle(image, tuple(point.astype(int)), radius=3, color=path_color(), thickness=-1)        
        
        # Save the image
        cv2.imwrite(os.path.join(output_directory, f'path_{idx}.jpg'), image)

# Main execution
if __name__ == "__main__":
    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged'  # Replace with the path to your JSON files
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b128_e400_v17.pth'
    num_samples = 500
    configurations = load_keypoints_from_truncated_json(directory)
    model = load_model_for_inference(model_path)
    # Parameters for PRM
    num_neighbors = 10  # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    roadmap, tree = build_lazy_roadmap_with_kdtree(configurations, num_neighbors, model)   
    end_time = time.time()

    print("time taken to find the graph", end_time - start_time)  

    # Define start and goal configurations as numpy arrays
    # start_config = np.array([[255, 441], [258, 311], [201, 300], [144, 290], [150, 260], [144, 191], [136, 120], [112, 103], [133, 73]])
    # goal_config = np.array([[250, 442], [252, 311], [260, 252], [267, 193], [298, 196], [373, 203], [448, 209], [483, 205], [487, 249]])

    start_config = np.array([[250, 442], [252, 311], [275, 255], [294, 201], [323, 209], [368, 268], [411, 328], [443, 343], [426, 382]])
    goal_config = np.array([[250, 442], [252, 311], [210, 271], [167, 231], [188, 209], [227, 147], [265, 85], [278, 56], [315, 73]])

    # Add start and goal configurations to the roadmap
    start_node, tree = add_config_to_roadmap(start_config, roadmap, tree, num_neighbors)
    goal_node, tree = add_config_to_roadmap(goal_config, roadmap, tree, num_neighbors)
        
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
         print("goal sets: ", goal_sets)
    
         with open("config/dl_multi_features.yaml", "w") as yaml_file:
             s = "dl_controller:\n"
             s += "  num_goal_sets: " + str(len(goal_sets)) + "\n"
             for i, goal in enumerate(goal_sets, start=1):
                 # Convert the list of floats into a comma-separated string
                 goal_str = ', '.join(map(str, goal))
                 s += f"  goal_features{i}: [{goal_str}]\n"
    
             # Write the string to the file
             yaml_file.write(s)
    
         print("Data successfully written to config/dl_multi_features.yaml")

         with open("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/1/dl_multi_features.yaml", "w") as yaml_file:
             s = "dl_controller:\n"
             s += "  num_goal_sets: " + str(len(goal_sets)) + "\n"
             for i, goal in enumerate(goal_sets, start=1):
                 # Convert the list of floats into a comma-separated string
                 goal_str = ', '.join(map(str, goal))
                 s += f"  goal_features{i}: [{goal_str}]\n"
    
             # Write the string to the file
             yaml_file.write(s)

         # Save configurations to a .txt file
         with open("config/path_configurations_no_obs.txt", "w") as file:
             file.write("Start Configuration:\n")
             file.write(str(start_config.tolist()) + "\n\n")
             file.write("Goal Configuration:\n")
             file.write(str(goal_config.tolist()) + "\n\n")
             file.write("Path:\n")
             for config in path:
                 file.write(str(config.tolist()) + "\n")
             file.write("\nPoint Set:\n")
             for points in point_set:
                 file.write(str(points) + "\n")

         with open("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/1/path_configurations_no_obs.txt", "w") as file:
             file.write("Start Configuration:\n")
             file.write(str(start_config.tolist()) + "\n\n")
             file.write("Goal Configuration:\n")
             file.write(str(goal_config.tolist()) + "\n\n")
             file.write("Path:\n")
             for config in path:
                 file.write(str(config.tolist()) + "\n")
             file.write("\nPoint Set:\n")
             for points in point_set:
                 file.write(str(points) + "\n")

        

         print("Configurations successfully saved to configurations.txt")
    # print("time taken to find the graph", end_time - start_time)  

    

