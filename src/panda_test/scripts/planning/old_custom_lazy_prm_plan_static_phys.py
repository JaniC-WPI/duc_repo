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
SAFE_DISTANCE = 100  # Safe distance from the obstacle

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

    distance = np.linalg.norm(output, axis=-1)
    return float(distance)  # Reshape to the original configuration format

# def calculate_model_distances(configurations, model):
#     num_configs = len(configurations)
#     distance_matrix = np.zeros((num_configs, num_configs))
    
#     for i in range(num_configs):
#     #   distance_shape_check = predict_custome_distance(configurations[i], configurations[i+1], model)
#     #   print(distance_shape_check)
#       for j in range(num_configs):
#           if i!=j:
#             distance_matrix = predict_custom_distance(configurations[i], configurations[j], model)
#             print(distance_matrix)

#     return distance_matrix

# def predict_custom_distance(start_kps, next_kps, model):
#     # Convert to 2D tensors if necessary
#     start_kp_tensor = torch.tensor(start_kps, dtype=torch.float)  
#     next_kp_tensor = torch.tensor(next_kps, dtype=torch.float) 

#     # Check if CUDA is available and move tensors to GPU if possible
#     if torch.cuda.is_available():
#         start_kp_tensor = start_kp_tensor.cuda()
#         next_kp_tensor = next_kp_tensor.cuda()
#         model = model.cuda()

#     # Predict the next configurations (in batch)
#     with torch.no_grad():
#         output = model(start_kp_tensor, next_kp_tensor).cpu().numpy()  

#     # Calculate distances (replace with your actual logic based on model's output)
#     distances = np.linalg.norm(output)  
#     return distances 

def calculate_model_distances(configurations, model, batch_size=32):
    num_configs = len(configurations)
    distance_matrix = np.zeros((num_configs, num_configs))
    
    # Convert configurations to a suitable format (flattened and batched) for the model
    configurations_flat = [config.flatten() for config in configurations]
    configurations_batch = torch.tensor(configurations_flat, dtype=torch.float)
    
    # Calculate distances in batches to optimize performance
    with torch.no_grad():  # Ensure gradient computation is disabled
        for i in range(0, num_configs, batch_size):
            batch_start = configurations_batch[i:i+batch_size]
            for j in range(0, num_configs, batch_size):
                batch_next = configurations_batch[j:j+batch_size]
                
                # Predict distances for all pairs in the current batch
                distances_batch = predict_custom_distance(batch_start, batch_next, model)
                
                # Compute the Euclidean norm for each pair to get scalar distances
                distances = np.linalg.norm(distances_batch, axis=-1)
                
                # Assign distances to the appropriate positions in the distance matrix
                end_i = min(i + batch_size, num_configs)
                end_j = min(j + batch_size, num_configs)
                distance_matrix[i:end_i, j:end_j] = distances[:end_i-i, :end_j-j]
    print(distance_matrix)
    return distance_matrix

# def calculate_model_distances(configurations, model):
#     num_configs = len(configurations)

#     # Reshape to allow broadcasting
#     configurations_array = np.array(configurations)
#     configs_a = configurations_array[:, None, :]  
#     configs_b = configurations_array[None, :, :]

#     # Calculate pairwise differences 
#     differences = configs_a - configs_b 

#     print("Shape of configurations_array:", configurations_array.shape)
#     print("Shape of differences:", differences.shape) 

#     # Reshape differences for batched input
#     start_kps = differences[:, :, :, 0].reshape(-1, 12)  # Assuming 12 is the flattened keypoint size
#     next_kps = differences[:, :, :, 1].reshape(-1, 12)

#     # Apply your model to the differences in batches (adjust batch size if needed)
#     all_distances = []
#     BATCH_SIZE = 5000  # Adjust as needed for your memory constraints
#     for i in range(0, len(start_kps), BATCH_SIZE):
#         batch_distances = predict_custom_distance(start_kps[i: i + BATCH_SIZE], next_kps[i: i + BATCH_SIZE], model)
#         all_distances.append(batch_distances)

#     distances = np.concatenate(all_distances)
#     print("Length of distances array:", len(distances))
#     distances = distances.reshape(num_configs, num_configs)  # Reshape into the distance matrix    

#     return distances


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


# def heuristic(config1, config2):
#     distance = vectorized_sum_pairwise_euclidean(config1, config2)
#     return distance * 0.8  # Slightly underestimate the distance

# Function to add a configuration to the roadmap with collision checking
def add_config_to_roadmap(config, G, tree, k_neighbors, obstacle_center, safe_distance, half_diagonal):
    print("Shape of config being added:", config.shape)
    flattened_config = config.flatten().reshape(1, -1)
    _, indices = tree.query(flattened_config, k=k_neighbors)
    connections = 0
    
    node_id = len(G.nodes)
    G.add_node(node_id, configuration=config)
    
    for i in indices[0]:
        neighbor_config = G.nodes[i]['configuration']
        print("Shape of neighbor_config:", neighbor_config.shape)
        # Here we need to convert configurations back to their original shape for collision checking
        if is_collision_free(np.vstack((config, neighbor_config)), obstacle_center, safe_distance, half_diagonal):
            G.add_edge(node_id, i)
            connections += 1

    if connections == 0:  # If no connections were made, remove the node
        G.remove_node(node_id)
        return None, tree
    
    # Update the tree with the new node for future queries
    new_flattened_configs = np.vstack([tree.data, flattened_config])
    new_tree = KDTree(new_flattened_configs)
    
    return node_id, new_tree

def distance_line_to_point(p1, p2, point):
    """Calculates the distance of a point to a line segment."""
    numerator = np.abs((p2[1] - p1[1]) * point[0] - (p2[0] - p1[0]) * point[1] + p2[0] * p1[1] - p2[1] * p1[0])
    denominator = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    return numerator / denominator

def square_obstacle(center, half_diagonal):
    """Creates a Shapely square polygon representing the obstacle."""
    dx = dy = half_diagonal
    x0, y0 = center[0] - dx, center[1] - dy  # Bottom-left corner
    return geom.Polygon(((x0, y0), (x0 + 2*dx, y0), (x0 + 2*dx, y0 + 2*dy), (x0, y0 + 2*dy)))

def is_collision_free(configuration, obstacle_center, safe_distance, half_diagonal):
    obstacle = square_obstacle(obstacle_center, half_diagonal + safe_distance)

    for i in range(len(configuration) - 1):
        line_segment = geom.LineString([configuration[i], configuration[i + 1]])
        if line_segment.distance(obstacle) <= 0:  # Collision! 
            return False

    return True  # Collision-free

def find_path(G, start_node, goal_node):
    """Find a path from start to goal in the roadmap G."""
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
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b64_e200_v1.pth'
    num_samples = 500
    configurations = load_keypoints_from_json(directory)
    model = load_model_for_inference(model_path)
    # distance_matrix = calculate_model_distances(configurations, model)
    # distance_matrix = np.array([1.0]).reshape(-1,1)
    # configurations = load_and_sample_configurations(directory, num_samples)
    # Parameters for PRM
    num_neighbors = 500 # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    roadmap, tree = build_lazy_roadmap_with_kdtree(configurations, num_neighbors, model)   
    end_time = time.time()

    # print("time taken to find the graph", end_time - start_time)  

    # Define start and goal configurations as numpy arrays
    start_config = np.array([[272, 437], [266, 314], [175, 261], [187, 236], [230, 108], [215, 85]]) 
    goal_config = np.array([[271, 436], [267, 313], [223, 213], [248, 199], [383, 169], [404, 147]]) 
    
    obstacle_center = (380, 33)
    half_diagonal = 20
    safe_distance = half_diagonal + SAFE_DISTANCE 

    # Add start and goal configurations to the roadmap
    start_node, tree = add_config_to_roadmap(start_config, roadmap, tree, num_neighbors, obstacle_center, safe_distance, half_diagonal)
    goal_node, tree = add_config_to_roadmap(goal_config, roadmap, tree, num_neighbors, obstacle_center, safe_distance, half_diagonal)
        
    # Find and print the path from start to goal
    path = find_path(roadmap, start_node, goal_node)

    output_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/physical_path_planning/scenarios/phys_path_scene_11'

    image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/physical_path_planning/scenarios/obstacle_image_11.png'

    if path:
        print("Path found:", path)
        plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)
    else:
        print("No path found")

