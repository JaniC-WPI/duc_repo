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

def load_model_for_inference(model_path):    
    model = PosRegModel(12)
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

def visualize_interactions(config1, config2, obstacle_boundary):
    fig, ax = plt.subplots()
    # Plot obstacle boundary
    obstacle_patch = PolygonPatch(obstacle_boundary, alpha=0.5, color="red", zorder=2)
    ax.add_patch(obstacle_patch)
    
    # Set plot limits and aspect
    ax.set_xlim([0, IMAGE_WIDTH])
    ax.set_ylim([IMAGE_HEIGHT, 0])
    ax.set_aspect('equal')

    # Visualize interactions within each configuration
    for config in [config1, config2]:
        for i in range(len(config) - 1):
            x_values, y_values = zip(*config[i:i+2])
            ax.plot(x_values, y_values, "blue", linewidth=2, solid_capstyle='round', zorder=1)

    # Visualize interactions between corresponding keypoints across configurations
    for i in range(len(config1)):
        x_values = [config1[i][0], config2[i][0]]
        y_values = [config1[i][1], config2[i][1]]
        ax.plot(x_values, y_values, "green", linewidth=2, linestyle="--", zorder=1)

    plt.show()
 
def visualize_interactions_path(configurations, obstacle_boundary):
    fig, ax = plt.subplots()
    # Plot obstacle boundary
    ax.add_patch(PolygonPatch(obstacle_boundary, alpha=0.5, color="red", zorder=2))
    
    # Set plot limits and aspect
    ax.set_xlim([0, IMAGE_WIDTH])
    ax.set_ylim([IMAGE_HEIGHT, 0])
    ax.set_aspect('equal')

    # Visualize paths within configurations
    for config in configurations:
        for i in range(len(config) - 1):
            segment = [config[i], config[i + 1]]
            line = geom.LineString(segment)

            # Draw the line segment
            x, y = line.xy
            ax.plot(x, y, "blue", linewidth=2, solid_capstyle='round', zorder=1)

            # Highlight if the segment intersects the obstacle
            if line.intersects(obstacle_boundary):
                ax.plot(x, y, "red", linewidth=3, solid_capstyle='round', zorder=1)

    # Visualize connections between corresponding keypoints of consecutive configurations
    for i in range(len(configurations) - 1):
        for k in range(len(configurations[i])):
            start_point = configurations[i][k]
            end_point = configurations[i + 1][k]
            line = geom.LineString([start_point, end_point])

            # Draw the connection line
            x, y = line.xy
            ax.plot(x, y, "green", linewidth=1, linestyle='--', zorder=1)

            # Highlight if the connection intersects the obstacle
            if line.intersects(obstacle_boundary):
                ax.plot(x, y, "orange", linewidth=2, linestyle='--', zorder=1)

    plt.show()

class CustomDistanceHeuristic:
    def __init__(self, model, graph):
        self.model = model
        self.graph = graph

    def __call__(self, current_node, target_node):
        current_config = self.graph.nodes[current_node]['configuration']
        target_config = self.graph.nodes[target_node]['configuration']
        distance = predict_custom_distance(current_config, target_config, self.model)
        return distance

def build_lazy_roadmap_with_kdtree(configurations, k_neighbors, model):
    """
    Build a LazyPRM roadmap using a KDTree for efficient nearest neighbor search.
    
    Args:
    - configurations: List[np.array], a list of configurations (keypoints flattened).
    - k_neighbors: int, the number of neighbors to connect to each node.
    
    Returns:
    - G: nx.Graph, the constructed roadmap.
    """
    configurations = configurations[1:9000:10]
    # print("Shape of configurations before building the roadmap:", len(configurations), configurations[0].shape)

    flattened_configs = np.vstack([config.flatten() for config in configurations])
    tree = BallTree(flattened_configs, metric=lambda x, y: predict_custom_distance(x, y, model))
    # tree = KDTree(flattened_configs)
    print("tree is built")
    # tree = KDTree(distance_matrix) 

    G = nx.Graph()

    for i, config in enumerate(configurations):
        G.add_node(i, configuration=config)

    for i, config in enumerate(flattened_configs):
        _, indices = tree.query([config], k=k_neighbors + 1)  # +1 to include self in results
        #indices = tree.query_radius(config.reshape(1,-1), r=20,count_only=False) # +1 to include self in results

        for j in indices[0]:  # Skip self
            if j !=i:
                G.add_edge(i, j)

    return G, tree

def add_config_to_roadmap(config,  G, tree, k_neighbors):
    """Add a configuration to the roadmap, connecting it to its k nearest neighbors."""
    flattened_config = config.flatten().reshape(1, -1)
    _, indices = tree.query(flattened_config, k=k_neighbors)
    
    node_id = len(G.nodes)
    G.add_node(node_id, configuration=config)
    
    for i in indices[0]:
        G.add_edge(node_id, i)
    
    # Update the tree with the new node for future queries
    new_flattened_configs = np.vstack([tree.data, flattened_config])
    # new_tree = KDTree(new_flattened_configs)
    
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
                print("collision detected")
                # If any segment intersects, the configuration is not collision-free
                return False
        
    for i in range(len(configuration1)):
        segment = geom.LineString([configuration1[i], configuration2[i]])
        if segment.intersects(obstacle_boundary):
            print("edge collision detected")
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
            print(f"Removed invalid edge: {u} <-> {v}")

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
            cv2.circle(image, tuple(point.astype(int)), radius=6, color=path_color(), thickness=-1)        
        
        # Save the image
        cv2.imwrite(os.path.join(output_directory, f'path_{idx}.jpg'), image)

# Main execution
if __name__ == "__main__":
    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_planning_kprcnn/'  # Replace with the path to your JSON files
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b64_e400_v6.pth'
    num_samples = 500
    configurations = load_keypoints_from_json(directory)
    model = load_model_for_inference(model_path)
    # distance_matrix = calculate_model_distances(configurations, model)
    # distance_matrix = np.array([1.0]).reshape(-1,1)
    # configurations = load_and_sample_configurations(directory, num_samples)
    # Parameters for PRM
    num_neighbors = 50 # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    roadmap, tree = build_lazy_roadmap_with_kdtree(configurations, num_neighbors, model)   
    end_time = time.time()

    print("time taken to find the graph", end_time - start_time)      

    # Define start and goal configurations as numpy arrays
    start_config = np.array([[267, 432], [269, 315], [200, 237], [219, 217], [322, 139], [344, 115]]) 
    goal_config = np.array([[267, 431], [269, 315], [240, 213], [266, 206], [387, 256], [417, 243]])

    SAFE_ZONE = 50  # Safe distance from the obstacle
    obstacle_center = (420, 133)
    half_diagonal = 20
    # safe_distance = SAFE_ZONE

    obstacle_boundary = geom.Polygon([
        (obstacle_center[0] - (half_diagonal + SAFE_ZONE), obstacle_center[1] - (half_diagonal + SAFE_ZONE)),
        (obstacle_center[0] + (half_diagonal + SAFE_ZONE), obstacle_center[1] - (half_diagonal + SAFE_ZONE)),
        (obstacle_center[0] + (half_diagonal + SAFE_ZONE), obstacle_center[1] + (half_diagonal + SAFE_ZONE)),
        (obstacle_center[0] - (half_diagonal + SAFE_ZONE), obstacle_center[1] + (half_diagonal + SAFE_ZONE)),
    ])

    # Add start and goal configurations to the roadmap
    # start_node = add_config_to_roadmap(start_config, roadmap, tree, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE)
    # goal_node = add_config_to_roadmap(goal_config, roadmap, tree, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE)
    start_node = add_config_to_roadmap(start_config, roadmap, tree, num_neighbors)
    goal_node = add_config_to_roadmap(goal_config, roadmap, tree, num_neighbors)

    validate_and_remove_invalid_edges(roadmap, obstacle_center, half_diagonal, SAFE_ZONE)
        
    # Find and print the path from start to goal
    path = find_path(roadmap, start_node, goal_node)
    # path = find_path_heuristic(roadmap, start_node, goal_node, heuristic)

    if path:
        point_set = []
        goal_sets = []
        # Iterate through the path, excluding the first and last configuration
        for configuration in path[1:-1]:
            # Extract the last three keypoints of each configuration
            last_three_points = configuration[-4:]
            last_three_points_float = [[float(point[0]), float(point[1])] for point in last_three_points]
            # Append these points to the point_set list
            point_set.append(last_three_points_float)
        # Iterate through the path, excluding start and goal
        for configuration in path[1:]: 
            last_three_points = configuration[-4:]
            last_three_points_float = [[float(point[0]), float(point[1])] for point in last_three_points]
            goal_features = []  # Create a new list for each goal set
            for point in last_three_points_float:
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

    
    

