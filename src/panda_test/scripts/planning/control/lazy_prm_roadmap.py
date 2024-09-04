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
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import pickle


# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Load keypoints from JSON files in a given directory
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
    return float(distance)  # Reshape to the original configuration format

def custom_distance(x, y):
    # Ensure x and y are in the format the model expects (flattened arrays)
    return predict_custom_distance(x, y, model)

def build_lazy_roadmap_with_kdtree(configurations, configuration_ids, joint_angles_dict, model, k_neighbors):
    """
    Build a LazyPRM roadmap using a KDTree for efficient nearest neighbor search.
    
    Args:
    - configurations: List[np.array], a list of configurations (keypoints flattened).
    - k_neighbors: int, the number of neighbors to connect to each node.
    
    Returns:
    - G: nx.Graph, the constructed roadmap.
    """

    flattened_configs = np.vstack([config.flatten() for config in configurations])
    # Extract joint angles using the configuration IDs
    joint_angles_list = [joint_angles_dict[config_id] for config_id in configuration_ids]
    joint_angles_array = np.vstack(joint_angles_list)

    custom_tree = BallTree(flattened_configs, metric=custom_distance)
    print("tree is built")

    custom_G = nx.Graph()

    for i, (config, config_id) in enumerate(zip(configurations, configuration_ids)):
        custom_G.add_node(i, configuration=config, joint_angles=joint_angles_dict[config_id])

    for i, config in enumerate(flattened_configs):
        distances, indices = custom_tree.query([config], k=k_neighbors + 1)  # +1 to include self in results

        for j,d in zip(indices[0], distances[0]):  # Skip self
            if j !=i:
                custom_G.add_edge(i, j, weight=d)     
        
    pos_dict = {n[0]:n[1]["configuration"][5] for n in custom_G.nodes.items()} 
    nx.draw_networkx(custom_G,node_size=5,width=0.3, with_labels=False, pos=pos_dict)
    # plt.show()  

    euclidean_tree = KDTree(flattened_configs)

    euclidean_G = nx.Graph()

    for i, (config, config_id) in enumerate(zip(configurations, configuration_ids)):
        euclidean_G.add_node(i, configuration=config, joint_angles=joint_angles_dict[config_id])

    for i, config in enumerate(flattened_configs):
        distances, indices = euclidean_tree.query([config], k=k_neighbors + 1)  # +1 to include self in results

        for j,d in zip(indices[0], distances[0]):  # Skip self
            if j !=i:
                euclidean_G.add_edge(i, j, weight=d)        

    return custom_G, custom_tree, euclidean_G, euclidean_tree

def save_graph(custom_graph, custom_tree, euclidean_graph, euclidean_tree, custom_graph_path, custom_tree_path, euclidean_g_path, euclidean_tree_path):
    with open(custom_graph_path, 'wb') as f:
        pickle.dump(custom_graph, f)
    with open(custom_tree_path, 'wb') as f:
        pickle.dump(custom_tree, f)
    print(f"Graph saved to {custom_graph_path}")
    print(f"BallTree saved to {custom_tree_path}")

    with open(euclidean_g_path, 'wb') as f:
        pickle.dump(euclidean_graph, f)
    with open(euclidean_tree_path, 'wb') as f:
        pickle.dump(euclidean_tree, f)
    print(f"Graph saved to {euclidean_g_path}")
    print(f"KDTree saved to {euclidean_tree_path}")

# Main execution
if __name__ == "__main__":
    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged/'  # Replace with the path to your JSON files
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b128_e500_v32.pth'
    custom_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_roadmap_angle.pkl'
    custom_tree_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_tree_angle.pkl'
    euclidean_g_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_roadmap_angle.pkl'
    euclidean_tree_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_tree_angle.pkl'

    configurations, configuration_ids = load_keypoints_from_json(directory)
    model = load_model_for_inference(model_path)
    joint_angles_dict = load_joint_angles_from_json(directory)

    skip_step = 10
    start_index = 1
    end_index = 25000

    skipped_configs, skipped_ids = skip_configurations(configurations, configuration_ids, \
                                                       skip_step, start_index, end_index)

    # Parameters for PRM
    num_neighbors = 25

     # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    custom_roadmap, custom_tree, euclidean_roadmap, euclidean_tree = build_lazy_roadmap_with_kdtree(\
                                skipped_configs, skipped_ids, joint_angles_dict, model, num_neighbors)   
    end_time = time.time()

    print("time taken to find the graph", end_time - start_time)      

    save_graph(custom_roadmap, custom_tree, euclidean_roadmap, euclidean_tree, \
               custom_graph_path, custom_tree_path, euclidean_g_path, euclidean_tree_path) 

    
    

