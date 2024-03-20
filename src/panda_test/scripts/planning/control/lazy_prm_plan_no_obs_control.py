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
import matplotlib.pyplot as plt

# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
SAFE_DISTANCE = 20  # Safe distance from the obstacle
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
    return configurations

def build_lazy_roadmap_with_kdtree(configurations, k_neighbors):
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
    tree = KDTree(flattened_configs)
    G = nx.Graph()
   # flattened_configs = flattened_configs[1:9000:10]
   # configurations = configurations[1:9000:10]

    for i, config in enumerate(configurations):
        G.add_node(i, configuration=config)

    for i, config in enumerate(flattened_configs):
        _, indices = tree.query([config], k=k_neighbors + 1)  # +1 to include self in results
        #indices = tree.query_radius(config.reshape(1,-1), r=20,count_only=False) # +1 to include self in results

        for j in indices[0]:  # Skip self
            if j!=i:
                G.add_edge(i, j)
        # for j in indices[0][1:]:  # Skip self
        #     distance = vectorized_sum_pairwise_euclidean(configurations[i], configurations[j]) # You'll need to define this distance calculation
        #     G.add_edge(i, j, weight=distance)
    # print(G.nodes.items())
    #SG= G.subgraph(range(1,9000,100))
    pos_dict = {n[0]:n[1]["configuration"][5] for n in G.nodes.items()} 
    nx.draw_networkx(G,node_size=5,width=0.3, with_labels=False, pos=pos_dict)
    plt.show()        
    return G, tree

# Add a new configuration to the roadmap
def add_config_to_roadmap(config, G, tree, k_neighbors=300):
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

# Find a path between start and goal configurations
def find_path(G, start_node, goal_node):
    """Find a path from start to goal in the roadmap G."""
    path_indices = nx.astar_path(G, source=start_node, target=goal_node)
    path_configurations = [G.nodes[i]['configuration'] for i in path_indices]
    return path_configurations

# Main execution
if __name__ == "__main__":
    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_planning_kprcnn/'  # Replace with the path to your JSON files
    num_samples = 300
    # configurations = load_and_sample_configurations(directory, num_samples)
    configurations = load_keypoints_from_json(directory)
    # Parameters for PRM
    num_neighbors = 10  # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    roadmap, tree = build_lazy_roadmap_with_kdtree(configurations, num_neighbors)   
    end_time = time.time()

    print("time taken to find the graph", end_time - start_time)  

    # Define start and goal configurations as numpy arrays
    start_config = np.array([[269, 431], [272, 315], [206, 232], [228, 214], [319, 122], [331, 93]]) 
    goal_config = np.array([[265, 430], [268, 314], [254, 209], [282, 205], [388, 284], [396, 314]])

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
             last_three_points = configuration[-5:]
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
    


