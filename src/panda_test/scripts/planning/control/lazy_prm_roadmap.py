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
import pickle, random


# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def log_config_joint_correspondence(configuration_ids, configurations, joint_angles_dict, output_path):
    with open(output_path, 'w') as f:
        f.write("ID, Configuration, Joint Angles\n")
        for config_id, config in zip(configuration_ids, configurations):
            joint_angles = joint_angles_dict[config_id]  # Fetch corresponding joint angles using the ID
            f.write(f"{config_id}, {config.tolist()}, {joint_angles.tolist()}\n")
    print(f"Configuration and Joint Angle correspondence logged to {output_path}")

def log_skipped_config_joint_correspondence(skipped_ids, skipped_configs, joint_angles_dict, output_path):
    with open(output_path, 'w') as f:
        f.write("ID, Configuration, Joint Angles\n")
        for config_id, config in zip(skipped_ids, skipped_configs):
            joint_angles = joint_angles_dict[config_id]  # Fetch corresponding joint angles using the ID
            f.write(f"{config_id}, {config.tolist()}, {joint_angles.tolist()}\n")
    print(f"Skipped Configuration and Joint Angle correspondence logged to {output_path}")


# Load keypoints from JSON files in a given directory
def load_keypoints_from_json(directory):
    configurations = []
    configuration_ids = []
    for filename in os.listdir(directory):
        if filename.endswith('.json') and not filename.endswith('_joint_angles.json') and not filename.endswith('_vel.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]  # Extracting x, y coordinates 
                           
                # Check if any keypoint's y component is greater than 390
                if any(kp[1] > 390 for kp in keypoints[2:]):  # Start checking from the third keypoint
                    print(f"Skipping configuration {data['id']} due to a keypoint (from third onward) with y > 390.")
                    continue  # Skip this configuration
                
                # If valid, add the configuration and its ID to the lists
                configurations.append(np.array(keypoints))
                configuration_ids.append(data['id'])  # Store the configuration ID

    print(f"Number of valid configurations: {len(configurations)}")
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

# def skip_configurations(configurations, configuration_ids, skip_step, start, end):
#     # Skip the configurations based on the step, start, and end parameters
#     skipped_configs = configurations[start:end:skip_step]
#     skipped_ids = configuration_ids[start:end:skip_step]
    
#     # Pair the skipped configurations with their IDs for consistent shuffling
#     paired_configs = list(zip(skipped_configs, skipped_ids))
    
#     # Shuffle the paired configurations and IDs
#     random.shuffle(paired_configs)
    
#     # Unzip the shuffled pairs back into configurations and IDs
#     shuffled_configs, shuffled_ids = zip(*paired_configs)

#     print("length of shuffled configurations", len(shuffled_configs))
    
#     return list(shuffled_configs), list(shuffled_ids)

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
    euclidean_tree = KDTree(flattened_configs)
    gt_tree = KDTree(joint_angles_array)

    custom_G = nx.Graph()
    euclidean_G = nx.Graph()
    gt_G = nx.Graph()

    # Add nodes in the graph from the configurations and 
    for i, (config, config_id) in enumerate(zip(configurations, configuration_ids)):
        custom_G.add_node(i, configuration=config, joint_angles=joint_angles_dict[config_id])
        euclidean_G.add_node(i, configuration=config, joint_angles=joint_angles_dict[config_id])
        gt_G.add_node(i, configuration=config, joint_angles=joint_angles_dict[config_id])        

    # generate nearest neighbors with custom joint distance
    for i, config in enumerate(flattened_configs):
        distances, indices = custom_tree.query([config], k=k_neighbors + 1)  # +1 to include self in results
        for j, d in zip(indices[0], distances[0]):  # Skip self
            if j != i:
                custom_G.add_edge(i, j, weight=d)

    # generate nearest neighbors with default euclidean distances between configs in image space
    for i, config in enumerate(flattened_configs):
        distances, indices = euclidean_tree.query([config], k=k_neighbors + 1)  # +1 to include self in results
        for j, d in zip(indices[0], distances[0]):  # Skip self
            if j != i:
                euclidean_G.add_edge(i, j, weight=d)

    # generate nearest neighbors with actual joint distances between configs in configuration space
    for i, joint_angles in enumerate(joint_angles_array):
        distances, indices = gt_tree.query([joint_angles], k=k_neighbors + 1)  # +1 to include the node itself        
        for d, j in zip(distances[0], indices[0]):
            if i != j:  # Avoid self-loops
                gt_G.add_edge(i, j, weight=d)  

    return custom_G, custom_tree, euclidean_G, euclidean_tree, gt_G, gt_tree

def save_graph(custom_graph, custom_tree, euclidean_graph, euclidean_tree, \
                gt_graph, gt_tree, custom_graph_path, custom_tree_path, \
                euclidean_g_path, euclidean_tree_path, gt_graph_path, gt_tree_path):
    
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

    with open(gt_graph_path, 'wb') as f:
        pickle.dump(gt_graph, f)
    with open(gt_tree_path, 'wb') as f:
        pickle.dump(gt_tree, f)
    print(f"Graph saved to {gt_graph_path}")
    print(f"KDTree with joint distance saved to {gt_tree_path}")

def verify_roadmap(roadmap, configurations, name="Roadmap"):
    """
    Verify that all connected configurations in the roadmap are part of the original configurations.

    Args:
    - roadmap: nx.Graph, the roadmap graph to check.
    - configurations: List[np.array], the original list of configurations loaded from keypoints.
    - name: str, the name of the roadmap for logging purposes.
    """
    print(f"\nVerifying {name}...")

    # Iterate over each edge in the roadmap and verify the configurations
    for edge in roadmap.edges():
        node1, node2 = edge
        config1 = roadmap.nodes[node1]['configuration']
        config2 = roadmap.nodes[node2]['configuration']

        # Check if configurations are part of the original dataset
        if not np.any([np.array_equal(config1, c) for c in configurations]):
            print(f"Mismatch found in {name}: Node {node1} configuration is not part of the original dataset.")
            break
        if not np.any([np.array_equal(config2, c) for c in configurations]):
            print(f"Mismatch found in {name}: Node {node2} configuration is not part of the original dataset.")
            break
        else:
            print(f"Edge between {node1} and {node2} in {name} is valid.")

    print(f"Verification complete for {name}.\n")

# Main execution
if __name__ == "__main__":
    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged/'  # Replace with the path to your JSON files
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b128_e500_v32.pth'
    custom_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_roadmap_angle_fresh.pkl'
    custom_tree_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_tree_angle_fresh.pkl'
    euclidean_g_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_roadmap_angle_fresh.pkl'
    euclidean_tree_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_tree_angle_fresh.pkl'
    gt_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_roadmap_angle_fresh.pkl'
    gt_tree_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_tree_angle_fresh.pkl'

    configurations, configuration_ids = load_keypoints_from_json(directory)
    model = load_model_for_inference(model_path)
    joint_angles_dict = load_joint_angles_from_json(directory)

    # correspondence_log_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/config_joint_correspondence.csv'
    # log_config_joint_correspondence(configuration_ids, configurations, joint_angles_dict, correspondence_log_path)

    skip_step = 10
    start_index = 1
    end_index = 25000

    skipped_configs, skipped_ids = skip_configurations(configurations, configuration_ids, \
                                                       skip_step, start_index, end_index)
    
    # skipped_log_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/skipped_config_joint_correspondence.csv'
    # log_skipped_config_joint_correspondence(skipped_ids, skipped_configs, joint_angles_dict, skipped_log_path)

    # Parameters for PRM
    num_neighbors = 25

     # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    custom_roadmap, custom_tree, euclidean_roadmap, euclidean_tree, gt_roadmap, gt_tree = build_lazy_roadmap_with_kdtree(\
                                skipped_configs, skipped_ids, joint_angles_dict, model, num_neighbors)   
    end_time = time.time()

    print("time taken to find the graph", end_time - start_time)      

    # verify_roadmap(custom_roadmap, configurations, name="Custom Roadmap")
    # verify_roadmap(euclidean_roadmap, configurations, name="Euclidean Roadmap")
    # verify_roadmap(gt_roadmap, configurations, name="Ground Truth Roadmap")
   
    save_graph(custom_roadmap, custom_tree, euclidean_roadmap, euclidean_tree, \
                gt_roadmap, gt_tree, custom_graph_path, custom_tree_path, \
                euclidean_g_path, euclidean_tree_path, gt_graph_path, gt_tree_path)

    
    

