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
import pickle, csv

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
        G.add_node(i, configuration=config, joint_angles=joint_angles_dict[config_id])

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
                G.add_edge(i, j, weight=d)  # Use joint displacement as edge weight
                 
                # Debugging statement to trace edge creation
                print(f"Edge added between Node {i} and Node {j} with joint displacement {joint_displacement} and joint distance {distance}")

    return G, tree

def save_graph(graph, tree, graph_path, tree_path):
    with open(graph_path, 'wb') as f:
        pickle.dump(graph, f)
    with open(tree_path, 'wb') as f:
        pickle.dump(tree, f)
    print(f"Graph saved to {graph_path}")
    print(f"BallTree saved to {tree_path}")

# Main execution
if __name__ == "__main__":
    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged/'  # Replace with the path to your JSON files
    graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_roadmap_angle.pkl'
    tree_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_tree_angle.pkl'
    
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

    save_graph(roadmap, tree, graph_path, tree_path)  
