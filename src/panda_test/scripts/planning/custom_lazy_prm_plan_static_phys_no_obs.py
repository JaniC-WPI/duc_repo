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
from pos_regression import PosRegModel

# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Load keypoints from JSON files in a given directory
def load_keypoints_from_json(directory):
    configurations = []
    for filename in os.listdir(directory):
        # if filename.endswith('.json'):
        if filename.endswith('.json') and not filename.endswith('_combined.json') and not filename.endswith('_vel.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                # Convert keypoints to integers
                keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]  # Extracting x, y coordinates
                configurations.append(np.array(keypoints))
    # print(configurations)
    print("Shape of a single configuration:", configurations[0].shape)  
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
    print("Shape of configurations before building the roadmap:", len(configurations), configurations[0].shape)

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
        
    print(G.nodes.items())
    #SG= G.subgraph(range(1,9000,100))
    pos_dict = {n[0]:n[1]["configuration"][5] for n in G.nodes.items()}      
    print(pos_dict) 
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
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/path_planning_panda_regression/'  # Replace with the path to your JSON files
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
    start_config = np.array([[272, 437], [266, 314], [175, 261], [187, 236], [230, 108], [215, 85]]) 
    goal_config = np.array([[271, 436], [267, 313], [223, 213], [248, 199], [383, 169], [404, 147]]) 
    
    # Add start and goal configurations to the roadmap
    start_node, tree = add_config_to_roadmap(start_config, roadmap, tree, num_neighbors)
    goal_node, tree = add_config_to_roadmap(goal_config, roadmap, tree, num_neighbors)
    
    heuristic = CustomDistanceHeuristic(model, roadmap)
    # Find and print the path from start to goal
    path = find_path_heuristic(roadmap, start_node, goal_node, heuristic)
    # path = find_path(roadmap, start_node, goal_node)

    output_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/physical_path_planning/scenarios/scenarios_custom/phys_path_no_obs_with_cost_scene_07/'

    image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/physical_path_planning/scenarios/obstacle_image_07.png'

    if path:
        print("Path found:", path)
        plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)
    else:
        print("No path found")

