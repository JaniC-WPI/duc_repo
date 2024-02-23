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
import torchvision.transforms as T
import yaml
import shapely.geometry as geom
import scipy


# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
SAFE_DISTANCE = 70  # Safe distance from the obstacle
# COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

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
    # print("Shape of a single configuration:", configurations[0].shape)  
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

def detect_green_rectangle(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range for green color
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate half of the diagonal using Pythagoras theorem
        half_diagonal = np.sqrt(w**2 + h**2) / 2
        return (int(x + w/2), int(y + h/2), int(half_diagonal))
    return None

# def sum_pairwise_euclidean(config1, config2):
#     total_distance = 0
#     for point1, point2 in zip(config1, config2):
#         total_distance += np.linalg.norm(point1 - point2)
#     return total_distance

def vectorized_sum_pairwise_euclidean(config1, config2):
    diffs = config1 - config2  # Calculate differences directly
    # print("Shape of diffs:", diffs.shape) # Add this line
    distances_squared = np.sum(diffs * diffs, axis=1)  # Adjust axis for summation 
    distances = np.sqrt(distances_squared)
    return np.sum(distances)  

def build_lazy_roadmap_with_kdtree(configurations, k_neighbors):
    """
    Build a LazyPRM roadmap using a KDTree for efficient nearest neighbor search.
    
    Args:
    - configurations: List[np.array], a list of configurations (keypoints flattened).
    - k_neighbors: int, the number of neighbors to connect to each node.
    
    Returns:
    - G: nx.Graph, the constructed roadmap.
    """
    # print("Shape of configurations before building the roadmap:", len(configurations), configurations[0].shape)

    flattened_configs = np.vstack([config.flatten() for config in configurations])
    tree = KDTree(flattened_configs)
    G = nx.Graph()
    
    for i, config in enumerate(configurations):
        G.add_node(i, configuration=config)
        
    for i, config in enumerate(flattened_configs):
        _, indices = tree.query([config], k=k_neighbors + 1)  # +1 to include self in results
        for j in indices[0][1:]:  # Skip self
            G.add_edge(i, j)
        # for j in indices[0][1:]:  # Skip self
            # distance = vectorized_sum_pairwise_euclidean(configurations[i], configurations[j]) # You'll need to define this distance calculation
            # G.add_edge(i, j, weight=distance)
                        
    return G, tree

def square_obstacle(center, half_diagonal):
    """Creates a Shapely square polygon representing the obstacle."""
    dx = dy = half_diagonal
    x0, y0 = center[0] - dx, center[1] - dy  # Bottom-left corner
    return geom.Polygon(((x0, y0), (x0 + 2*dx, y0), (x0 + 2*dx, y0 + 2*dy), (x0, y0 + 2*dy)))

def is_collision_free(configuration, obstacle_center, safe_distance, half_diagonal):
    # print("shape of configuration in collision", configuration.shape)
    obstacle = square_obstacle(obstacle_center, half_diagonal + safe_distance)

    for i in range(len(configuration) - 1):
        line_segment = geom.LineString([configuration[i], configuration[i + 1]])
        if line_segment.distance(obstacle) <= 0:  # Collision! 
            return False

    return True  # Collision-free

def rrt(roadmap, start_config, goal_config, obstacle_center, safe_distance, half_diagonal, max_iterations, step_size, kd_tree):
    tree = nx.Graph() 
    tree.add_node(0, configuration=start_config)  # Initialize tree with start

    for _ in range(max_iterations):
        rand_config = get_random_config(roadmap)  # Sample with bias towards PRM
        print("Type of rand_config after get_random_config:", type(rand_config), type(rand_config[0])) 
        # print("Random Config", rand_config)
        # print(type(rand_config))
        nearest_node_id, nearest_config = find_nearest_neighbor(tree, np.array(rand_config))
        new_config = extend(nearest_config, rand_config, step_size, roadmap, kd_tree)

        if is_collision_free(new_config, obstacle_center, safe_distance, half_diagonal):
            new_node_id = len(tree.nodes) 
            tree.add_node(new_node_id, configuration=new_config)
            tree.add_edge(nearest_node_id, new_node_id)

            if is_goal_reached(new_config, goal_config):
                return find_path_in_tree(tree, 0, new_node_id)  # Path found!

    return None  # No path found within iterations 

def get_random_config(roadmap):
    config = roadmap.nodes[np.random.choice(len(roadmap.nodes))]['configuration'] 

    print("Type of config inside get_random_config:", type(config), type(config[0]))  # Check element type
    return config
        
def find_nearest_neighbor(tree, config):
    # print("Shape of config inside find_nearest_neighbor:", config.shape) 
    tree_configs = np.array([node[1]['configuration'].flatten() for node in tree.nodes.items()])
    kdtree = KDTree(tree_configs)
    _, index = kdtree.query(config.reshape(1, -1))
    nearest_node_id = list(tree.nodes)[index[0][0]]
    return nearest_node_id, tree.nodes[nearest_node_id]['configuration'] 

def extend(config1, config2, step_size, roadmap, tree):
    direction = config2 - config1
    print("direction", direction)
    magnitude = np.linalg.norm(direction)
    print(magnitude)
    if magnitude == 0:
        unit_direction = 0
    else:
        unit_direction = direction / magnitude
    # partial_step = step_size / 2
    intermediate_config = config1 + unit_direction * min(step_size, magnitude)

    print("Config1", config1)
    print("Intermediate config", intermediate_config)

    # Use your KDTree to find the nearest neighbor in the roadmap
    _, index = tree.query(intermediate_config.reshape(1, -1)) 
    print(index)
    node_id = index[0][0]  # Extract the integer index
    new_config = roadmap.nodes[node_id]['configuration'] 
    
    return new_config

def is_goal_reached(config, goal_config, threshold=30): # Adjust threshold as needed
    distance = vectorized_sum_pairwise_euclidean(config, goal_config)
    return distance <= threshold

def find_path_in_tree(tree, start_node_id, goal_node_id):
    path_node_indices = nx.shortest_path(tree, source=start_node_id, target=goal_node_id)
    path_configurations = [tree.nodes[node_id]['configuration'] for node_id in path_node_indices]
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
    num_samples = 500
    configurations = load_keypoints_from_json(directory)
    # configurations = load_and_sample_configurations(directory, num_samples)
    # Parameters for PRM
    max_iterations = 5000 
    step_size = 5
    num_neighbors = 5000 # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    roadmap, tree = build_lazy_roadmap_with_kdtree(configurations, num_neighbors)   
    end_time = time.time()

    print("time taken to find the graph", end_time - start_time)  

    # Define start and goal configurations as numpy arrays
    start_config = np.array([[272, 437], [266, 314], [175, 261], [187, 236], [230, 108], [215, 85]]) 
    goal_config = np.array([[271, 436], [267, 313], [223, 213], [248, 199], [383, 169], [404, 147]]) 
    
    obstacle_center = (400, 53)
    half_diagonal = 20
    safe_distance = half_diagonal + SAFE_DISTANCE 

    # Add start and goal configurations to the roadmap
    # start_node, tree = add_config_to_roadmap(start_config, roadmap, tree, num_neighbors, obstacle_center, safe_distance, half_diagonal)
    # goal_node, tree = add_config_to_roadmap(goal_config, roadmap, tree, num_neighbors, obstacle_center, safe_distance, half_diagonal)
        
    # Find and print the path from start to goal

    path = rrt(roadmap, start_config, goal_config, obstacle_center, safe_distance, half_diagonal, max_iterations, step_size, tree)


    output_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/physical_path_planning/scenarios/phys_path_scene_05'

    image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/physical_path_planning/scenarios/obstacle_image_05.png'

    if path:
        print("Path found:", path)
        plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)
    else:
        print("No path found")

