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
import torchvision.transforms as T
import yaml
import shapely.geometry as geom
import scipy
import matplotlib.pyplot as plt
# from pos_regression import PosRegModel
from descartes import PolygonPatch
import torch
from matplotlib.font_manager import FontProperties

from pos_regression import PosRegModel

# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
SAFE_DISTANCE = 50  # Safe distance from the obstacle

def load_matched_configurations(directory):
    # Initialize empty lists for configurations
    kp_configurations = []
    jt_configurations = []
    identifiers = [] 

    # Temporary dictionary to hold joint angles keyed by identifier
    temp_jt_configurations = {}

    # First pass: Load joint angles into temporary dictionary
    for filename in (os.listdir(directory)):
        if filename.endswith('_joint_angles.json'):
            identifier = filename.replace('_joint_angles.json', '')
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                temp_jt_configurations[identifier] = np.array(data['joint_angles'])

    # Second pass: Match and load keypoints configurations
    for filename in (os.listdir(directory)):
        if filename.endswith('.json') and not filename.endswith('_joint_angles.json') and not filename.endswith('_vel.json'):
            identifier = filename.replace('.json', '')
            if identifier in temp_jt_configurations:
                with open(os.path.join(directory, filename), 'r') as file:
                    data = json.load(file)
                    keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]
                    kp_configurations.append(np.array(keypoints))
                    jt_configurations.append(temp_jt_configurations[identifier])
                    identifiers.append(identifier)

    return kp_configurations, jt_configurations, identifiers

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


def build_lazy_roadmap(kp_configurations, jt_configurations, k_neighbors, model):
    """
    Build a LazyPRM roadmap using a KDTree for efficient nearest neighbor search.
    
    Args:
    - configurations: List[np.array], a list of configurations (keypoints flattened).
    - k_neighbors: int, the number of neighbors to connect to each node.
    
    Returns:
    - G: nx.Graph, the constructed roadmap.
    """
    kp_configurations = kp_configurations[1:9000:10]
    jt_configurations = jt_configurations[1:9000:10]

    flattened_kp_configs = np.vstack([config.flatten() for config in kp_configurations])
    tree1 = KDTree(flattened_kp_configs)
    G1 = nx.Graph()
   
    for i, config in enumerate(kp_configurations):
        G1.add_node(i, configuration=config)

    for i, config in enumerate(flattened_kp_configs):
        dist, indices = tree1.query([config], k=k_neighbors + 1)
        for d, j in zip(dist[0], indices[0]):
            if j != i:
                G1.add_edge(i, j, distance=d)

    tree2 = BallTree(flattened_kp_configs, metric=lambda x, y: predict_custom_distance(x, y, model))
    G2 = nx.Graph()

    for i, config in enumerate(kp_configurations):
        G2.add_node(i, configuration=config)

    for i, config in enumerate(flattened_kp_configs):
        dist, indices = tree2.query([config], k=k_neighbors + 1)
        for d, j in zip(dist[0], indices[0]):
            if j != i:
                G2.add_edge(i, j, distance=d)

    flattened_jt_configs = np.vstack([config.flatten() for config in jt_configurations])
    tree3 = KDTree(flattened_jt_configs)
    G3 = nx.Graph()

    for i, config in enumerate(jt_configurations):
        G3.add_node(i, configuration=config)

    for i, config in enumerate(flattened_jt_configs):
        dist, indices = tree3.query([config], k=k_neighbors + 1)
        for d, j in zip(dist[0], indices[0]):
            if j != i:
                G3.add_edge(i, j, distance=d)     

    pos_dict = {n[0]:n[1]["configuration"][5] for n in G1.nodes.items()} 
    nx.draw_networkx(G1,node_size=5,width=0.3, with_labels=False, pos=pos_dict)
    plt.show()

    pos_dict = {n[0]:n[1]["configuration"][5] for n in G2.nodes.items()} 
    nx.draw_networkx(G2,node_size=5,width=0.3, with_labels=False, pos=pos_dict)
    plt.show()        

    # pos_dict = {n[0]:n[1]["configuration"][2] for n in G3.nodes.items()} 
    pos_dict = {n[0]:n[1]["configuration"][5] for n in G2.nodes.items()}
    nx.draw_networkx(G3,node_size=5,width=0.3, with_labels=False, pos=pos_dict)
    plt.show()

    return G1, G2, G3, tree1, tree2, tree3

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
                # If any segment intersects, the configuration is not collision-free
                return False
        
    for i in range(len(configuration1)):
        segment = geom.LineString([configuration1[i], configuration2[i]])
        if segment.intersects(obstacle_boundary):
            return False
        
     # If no segments intersect, the configuration is collision-free
    return True

def add_config_to_roadmap(config, G, tree, k_neighbors, obstacle_center, half_diagonal, safe_distance):
    # print("Shape of config being added:", config.shape)
    flattened_config = config.flatten().reshape(1, -1)
    _, indices = tree.query(flattened_config, k=k_neighbors)    
    node_id = len(G.nodes)
    
    G.add_node(node_id, configuration=config)
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
        # else:
        #     visualize_interactions(config, neighbor_config, obstacle_boundary)

    if nx.is_connected(G):
        print("Roadmap is connected")
    else:
        print("Roadmap is disconnected")  
    
    return node_id

def validate_and_remove_invalid_edges(G, obstacle_center, half_diagonal, safe_distance):
    # Iterate over a copy of the edges list to avoid modification issues during iteration
    for (u, v) in list(G.edges):
        config_u = G.nodes[u]['configuration']
        config_v = G.nodes[v]['configuration']
        # Perform the collision check for the edge
        if not is_collision_free(config_u, config_v, obstacle_center, half_diagonal, safe_distance):
            # If the edge is not collision-free, remove it from the graph
            G.remove_edge(u, v)

def find_path(G, start_node, goal_node):
    path_indices = nx.astar_path(G, source=start_node, target=goal_node)
    path_configurations = [G.nodes[i]['configuration'] for i in path_indices]

    return path_configurations, path_indices

def plot_all_joints(jt_configurations, G1, G2, G3):
    # Initialize lists to hold joint angles
    joint1_all = [config[0] for config in jt_configurations]
    joint2_all = [config[1] for config in jt_configurations]
    joint3_all = [config[2] for config in jt_configurations]

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting all configurations as points in 3D space
    ax.scatter(joint1_all, joint2_all, joint3_all, c='gray', marker='_')

    # Extract joint configurations from the graph nodes
    joint_configs = [G3.nodes[n]['configuration'] for n in G1.nodes()]
    joint1_euc = [config[0] for config in joint_configs]  # Joint 1 values
    joint2_euc = [config[1] for config in joint_configs]  # Joint 2 values
    joint3_euc = [config[2] for config in joint_configs]  # Joint 3 values

    print("length of joint configs in eucluidean space: ", len(joint_configs))

    ax.scatter(joint1_euc, joint2_euc, joint3_euc, c='r', marker='o')

    # Extract joint configurations from the graph nodes
    joint_configs = [G3.nodes[n]['configuration'] for n in G2.nodes()]
    joint1_cust = [config[0] for config in joint_configs]  # Joint 1 values
    joint2_cust = [config[1] for config in joint_configs]  # Joint 2 values
    joint3_cust= [config[2] for config in joint_configs]  # Joint 3 values

    print("length of joint configs in custom distance space: ", len(joint_configs))

    ax.scatter(joint1_cust, joint2_cust, joint3_cust, c='g', marker='o')

    # Setting labels for axes
    ax.set_xlabel('Joint 1 angle')
    ax.set_ylabel('Joint 2 angle')
    ax.set_zlabel('Joint 3 angle')

    # Display the plot
    plt.show()

def plot_common_node_pairs(G1,G2,G3):
    total_edges_g1 = G1.number_of_edges()
    total_edges_g2 = G2.number_of_edges()
    total_edges_g3 = G3.number_of_edges()

    # Find common connected edges between G1 and G3, and G2 and G3
    common_edges_g1_g3 = set(G1.edges()).intersection(set(G3.edges()))
    common_edges_g2_g3 = set(G2.edges()).intersection(set(G3.edges()))

    # Print the information
    print(f"Total connected edges in G1: {total_edges_g1}")
    print(f"Total connected edges in G2: {total_edges_g2}")
    print(f"Total connected edges in G3: {total_edges_g3}")
    print(f"Common connected edges between G1 and G3: {len(common_edges_g1_g3)}")
    print(f"Common connected edges between G2 and G3: {len(common_edges_g2_g3)}")
    
    common_pairs_g1_g3 = set(G1.edges()) & set(G3.edges())
    common_pairs_g2_g3 = set(G2.edges()) & set(G3.edges())

    common_pairs_g1_g3 = list(common_pairs_g1_g3)
    common_pairs_g2_g3 = list(common_pairs_g2_g3)
    # Unpack node pairs for G1 & G3
    x1, y1 = zip(*common_pairs_g1_g3) if common_pairs_g1_g3 else ([], [])
    # Unpack node pairs for G2 & G3
    x2, y2 = zip(*common_pairs_g2_g3) if common_pairs_g2_g3 else ([], [])
    
    plt.figure(figsize=(10, 5))

    plt.subplot(2,1,1)
    plt.scatter(x1, y1, color='blue', label=f'Common connected nodes between Euclidean and Baseline roadmap: {len(common_pairs_g1_g3)}', alpha=0.5)
    plt.legend() 
    plt.grid()
    plt.ylabel('Node Index 2', fontsize='medium')

    plt.subplot(2,1,2)
    plt.scatter(x2, y2, color='red', label=f'Common connected nodes between Custom Distance and Baseline roadmap:  {len(common_pairs_g2_g3)}', alpha=0.5)
    plt.legend() 
    plt.grid()

    plt.xlabel('Node Index 1', fontsize='medium')
    plt.ylabel('Node Index 2', fontsize='medium')
    plt.title('Common Node Pairs between Graphs', fontsize='large', fontweight='bold')  
   

    plt.tight_layout()
    plt.show()


def compare_edge_distances_scatter(G1, G2, G3):
    """
    Visualize comparison between edge distances of G1, G2 and G3 with euclidean distances of G3 in scatterplot
    """
    distances_g1 = []
    distances_g2 = []
    distances_g3 = []
    euclidean_distances_g13 = []
    euclidean_distances_g23 = []
    euclidean_distances_g33 = []
    
    for (u,v,d) in G1.edges(data=True):
        if G3.has_node(u) and G3.has_node(v):            
            pos_u = G3.nodes[u]['configuration']
            pos_v = G3.nodes[v]['configuration']
            euclidean_distance = calculate_euclidean_distance_3d(pos_u, pos_v)
            distances_g1.append(d['distance'])
            euclidean_distances_g13.append(euclidean_distance)

    for (u,v,d) in G2.edges(data=True):
        if G3.has_node(u) and G3.has_node(v):            
            pos_u = G3.nodes[u]['configuration']
            pos_v = G3.nodes[v]['configuration']
            euclidean_distance = calculate_euclidean_distance_3d(pos_u, pos_v)
            distances_g2.append(d['distance'])
            euclidean_distances_g23.append(euclidean_distance)

    for (u,v,d) in G3.edges(data=True):
        if G3.has_node(u) and G3.has_node(v):            
            pos_u = G3.nodes[u]['configuration']
            pos_v = G3.nodes[v]['configuration']
            euclidean_distance = calculate_euclidean_distance_3d(pos_u, pos_v)
            distances_g3.append(d['distance'])
            euclidean_distances_g33.append(euclidean_distance)
    
    # Scatter plot for distance comparison
    plt.scatter(euclidean_distances_g13, distances_g1, alpha=0.5)
    plt.xlabel('Euclidean distances in G3')
    plt.ylabel('Connected Edge disatnces in G1')
    plt.title('Scatter Plot of Distances in joint_space vs. kdtree_default')
    plt.grid(True)
    max_distance = max(max(euclidean_distances_g13), max(distances_g1))
    plt.plot([0, max(euclidean_distances_g13)], [0, max(distances_g1)], 'r--')  # y=x reference line
    plt.show()

    plt.scatter(euclidean_distances_g23, distances_g2, alpha=0.5)
    plt.xlabel('Euclidean distances in G3')
    plt.ylabel('Connected Edge disatnces in G2')
    plt.title('Scatter Plot of Distances in joint_space vs. custom_joint_space')
    plt.grid(True)
    max_distance = max(max(euclidean_distances_g23), max(distances_g2))
    print('max_euclidean_distances_g23', max(euclidean_distances_g23))
    print('max distances_g2', max(distances_g2))
    plt.plot([0, max(euclidean_distances_g23)], [0, max(distances_g2)], 'r--')  # y=x reference line
    plt.show()

    plt.scatter(euclidean_distances_g33, distances_g3, alpha=0.5)
    plt.xlabel('Euclidean distances in G3')
    plt.ylabel('Connected Edge disatnces in G3')
    plt.title('Scatter Plot of Distances in joint_space vs. actual_joint_space_disatnce')
    plt.grid(True)
    max_distance = max(max(euclidean_distances_g33), max(distances_g3))
    plt.plot([0, max(euclidean_distances_g33)], [0, max(distances_g3)], 'r--')  # y=x reference line
    plt.show()

def compare_edge_distances_hist(G1,G2,G3):
    distances_g1 = []
    distances_g2 = []
    distances_g3 = []
    euclidean_distances_g13 = []
    euclidean_distances_g23 = []
    euclidean_distances_g33 = []
    
    for (u,v,d) in G1.edges(data=True):
        if G3.has_node(u) and G3.has_node(v):            
            pos_u = G3.nodes[u]['configuration']
            pos_v = G3.nodes[v]['configuration']
            euclidean_distance = calculate_euclidean_distance_3d(pos_u, pos_v)
            distances_g1.append(d['distance'])
            euclidean_distances_g13.append(euclidean_distance)

    for (u,v,d) in G2.edges(data=True):
        if G3.has_node(u) and G3.has_node(v):            
            pos_u = G3.nodes[u]['configuration']
            pos_v = G3.nodes[v]['configuration']
            euclidean_distance = calculate_euclidean_distance_3d(pos_u, pos_v)
            distances_g2.append(d['distance'])
            euclidean_distances_g23.append(euclidean_distance)

    for (u,v,d) in G3.edges(data=True):
        if G3.has_node(u) and G3.has_node(v):            
            pos_u = G3.nodes[u]['configuration']
            pos_v = G3.nodes[v]['configuration']
            euclidean_distance = calculate_euclidean_distance_3d(pos_u, pos_v)
            distances_g3.append(d['distance'])
            euclidean_distances_g33.append(euclidean_distance)

    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.titleweight'] = 'bold'

    # Plot histograms
    fig, axs = plt.subplots(2, 3, figsize=(14, 10), sharey=True)

    x_limit = (0, 3.0)

    # label='Euclidean edge distance\n'
    # label='Custom joint space edge distance in\n keypoints configurations space'
    # label='Baseline euclidean distance\n'
    # label='Baseline euclidean distance\n for connected nodes in custom joint space'
    # label='Euclidean edge distance\n in actual joint space'
    # label='Baseline euclidean distance in actual joint space'
    
    axs[0, 0].hist(distances_g1, bins=30, alpha=0.7, label='1-a', color='blue')
    axs[0, 0].set_title('Euclidean Edge Distances')
    axs[1, 0].set_xlim(-0.05,3.0)

    axs[1, 0].hist(euclidean_distances_g13, bins=30, alpha=0.7, label='1-b', color='blue')
    axs[1, 0].set_title('Baseline Euclidean Distances')
    axs[1, 0].set_xlim(-0.05, 3.0)

    axs[0, 1].hist(distances_g2, bins=30, alpha=0.7, label='2-a', color='green')
    axs[0, 1].set_title('Custom Edge Distances')
    axs[1, 0].set_xlim(-0.05, 3.0)

    axs[1, 1].hist(euclidean_distances_g23, bins=30, alpha=0.7, label='2-b', color='green')
    axs[1, 1].set_title('Baseline Euclidean Distances')
    axs[1, 0].set_xlim(-0.05, 3.0)

    axs[0, 2].hist(distances_g3, bins=30, alpha=0.7, label='3-a', color='orange')
    axs[0, 2].set_title('Baseline Edge Distances')
    axs[1, 0].set_xlim(-0.05, 3.0)

    axs[1, 2].hist(euclidean_distances_g33, bins=30, alpha=0.7, label='3-b', color='orange')
    axs[1, 2].set_title('Baseline Euclidean Distances')
    axs[1, 0].set_xlim(-0.05, 3.0)

    font_prop = FontProperties(size='large', weight='bold')

    for ax in axs.flat:
        ax.label_outer()  # Hide x labels and tick labels for top plots and y ticks for right plots.
        ax.legend(prop=font_prop)
    
    plt.show()

def calculate_euclidean_distance_3d(pos_u, pos_v):
    """Calculate the Euclidean distance between two 3D positions."""
    return np.linalg.norm(np.array(pos_u) - np.array(pos_v))

def find_closest_configuration(target_config, kp_configurations):
    """Find the closest configuration to the target from a list of configurations."""
    # Flatten the target configuration for distance calculation
    target_flattened = target_config.flatten().reshape(1, -1)
    distances = np.linalg.norm(np.vstack([config.flatten() for config in kp_configurations]) - target_flattened, axis=1)
    closest_index = np.argmin(distances)
    return closest_index

def convert_configs(kp_configurations, jt_configurations, start_config, goal_config):
    # After loading configurations, find the closest configurations for start and goal
    closest_start_index = find_closest_configuration(start_config, kp_configurations)
    closest_goal_index = find_closest_configuration(goal_config, kp_configurations)
    
    start_joints = jt_configurations[closest_start_index]
    goal_joints = jt_configurations[closest_goal_index]
    
    return start_joints, goal_joints

def retrieve_joint_configs(path_indices, G3):
    intermediate_indices = path_indices[1:-1]  # This slices out the start and end nodes
    joint_configs = np.array([G3.nodes[idx]['configuration'] for idx in path_indices])
    return joint_configs

# def plot_joint_configs(joint_configs):
#     fig, axes = plt.subplots(3, 1, figsize=(10, 6))
#     for i in range(3):  # Assuming 3 joints
#         axes[i].plot(joint_configs[:, i], '-o', label=f'Joint {i+1}')
#         axes[i].set_xlabel('Step')
#         axes[i].set_ylabel('Angle')
#         axes[i].legend()
#     plt.tight_layout()
#     plt.show()

def plot_joint_configs(start_joints, goal_joints, intermediate_joint_configs1, intermediate_joint_configs2):
    """
    Plots the joint configurations of two intermediate paths, including the start and goal configurations.

    Parameters:
        start_joints (np.array): Joint configuration of the start position.
        goal_joints (np.array): Joint configuration of the goal position.
        intermediate_joint_configs1 (np.array): Array of joint configurations for the first intermediate path.
        intermediate_joint_configs2 (np.array): Array of joint configurations for the second intermediate path.
    """
    # Prepend start and append goal configurations to both paths
    g1_joint_configs = np.vstack([start_joints, intermediate_joint_configs1, goal_joints]) if intermediate_joint_configs1.size > 0 else np.vstack([start_joints, goal_joints])
    g2_joint_configs = np.vstack([start_joints, intermediate_joint_configs2, goal_joints]) if intermediate_joint_configs2.size > 0 else np.vstack([start_joints, goal_joints])

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    colors = ['blue', 'green']  # Define colors for the two paths
    labels = ['Path 1', 'Path 2']

    for idx, joint_configs in enumerate([g1_joint_configs, g2_joint_configs]):
        joint1_configs = joint_configs[:, 0]
        joint2_configs = joint_configs[:, 1]
        joint3_configs = joint_configs[:, 2]

        # Joint 1
        axs[0].plot(joint1_configs, '-o', color=colors[idx], label=f'Joint 1 {labels[idx]}')
        axs[0].set_title('Joint 1 Configuration over Paths')
        axs[0].set_xlabel('Step')
        axs[0].set_ylabel('Angle')

        # Joint 2
        axs[1].plot(joint2_configs, '-o', color=colors[idx], label=f'Joint 2 {labels[idx]}')
        axs[1].set_title('Joint 2 Configuration over Paths')
        axs[1].set_xlabel('Step')
        axs[1].set_ylabel('Angle')

        # Joint 3
        axs[2].plot(joint3_configs, '-o', color=colors[idx], label=f'Joint 3 {labels[idx]}')
        axs[2].set_title('Joint 3 Configuration over Paths')
        axs[2].set_xlabel('Step')
        axs[2].set_ylabel('Angle')

    for ax in axs:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

# def plot_joint_configs(start_joints, goal_joints, intermediate_joint_configs1, intermediate_joint_configs2):
#     """
#     Plots the joint configurations of the intermediate path, including the start and goal configurations.

#     Parameters:
#         start_joints (np.array): Joint configuration of the start position.
#         goal_joints (np.array): Joint configuration of the goal position.
#         intermediate_joint_configs (np.array): Array of joint configurations for the intermediate path.
#     """
#     # If there are intermediate configurations, prepend start and append goal configurations
#     if intermediate_joint_configs1.size > 0:
#         g1_joint_configs = np.vstack([start_joints, intermediate_joint_configs1, goal_joints])
#     else:
#         # If there are no intermediate steps (direct path), just plot start and goal
#         g1_joint_configs = np.vstack([start_joints, goal_joints])

#     if intermediate_joint_configs1.size > 0:
#         g2_joint_configs = np.vstack([start_joints, intermediate_joint_configs1, goal_joints])
#     else:
#         # If there are no intermediate steps (direct path), just plot start and goal
#         g2_joint_configs = np.vstack([start_joints, goal_joints])
#     # Separate joint configurations for plotting
#     joint1_configs = joint_configs[:, 0]
#     joint2_configs = joint_configs[:, 1]
#     joint3_configs = joint_configs[:, 2]

#     # Plotting
#     fig, axs = plt.subplots(3, 1, figsize=(10, 15))

#     # Joint 1
#     axs[0].plot(joint1_configs, '-o', label='Joint 1')
#     axs[0].set_title('Joint 1 Configuration over Path')
#     axs[0].set_xlabel('Step')
#     axs[0].set_ylabel('Angle')

#     # Joint 2
#     axs[1].plot(joint2_configs, '-o', label='Joint 2')
#     axs[1].set_title('Joint 2 Configuration over Path')
#     axs[1].set_xlabel('Step')
#     axs[1].set_ylabel('Angle')

#     # Joint 3
#     axs[2].plot(joint3_configs, '-o', label='Joint 3')
#     axs[2].set_title('Joint 3 Configuration over Path')
#     axs[2].set_xlabel('Step')
#     axs[2].set_ylabel('Angle')

#     for ax in axs:
#         ax.grid(True)
#         ax.legend()

#     plt.tight_layout()
#     plt.show()


# Main execution
if __name__ == "__main__":
    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_planning_kprcnn/' 
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b64_e400_v6.pth'    
    kp_configurations, joint_angles, identifiers = load_matched_configurations(directory)    
    model = load_model_for_inference(model_path)    
    num_neighbors = 25 # Number of neighbors for each node in the roadmap
    # Build the roadmaps
    roadmap1, roadmap2, roadmap3, tree1, tree2, tree3 = build_lazy_roadmap(kp_configurations, joint_angles, num_neighbors, model)   

    compare_edge_distances_hist(roadmap1,roadmap2,roadmap3)
    plot_common_node_pairs(roadmap1,roadmap2,roadmap3)

    # start_config = np.array([[269, 431], [272, 315], [172, 287], [178, 262], [149, 139], [119, 130]])  
    # goal_config = np.array([[263, 430], [268, 314], [257, 209], [284, 205], [399, 273], [414, 299]])

    # SAFE_ZONE = 50 
    # obstacle_center = (420, 103)
    # half_diagonal = 20

    # start_angle, goal_angle = convert_configs(kp_configurations, joint_angles, start_config, goal_config)

    # print(start_angle, goal_angle)

    # start_node_g1 = add_config_to_roadmap(start_config, roadmap1, tree1, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE)
    # goal_node_g1 = add_config_to_roadmap(goal_config, roadmap1, tree1, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE)

    # start_node_g2 = add_config_to_roadmap(start_config, roadmap2, tree2, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE)
    # goal_node_g2 = add_config_to_roadmap(goal_config, roadmap2, tree2, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE)

    # print(start_node_g2, goal_node_g2)
    # # start_node, tree = add_config_to_roadmap(start_config, roadmap, tree, 50)
    # # goal_node, tree = add_config_to_roadmap(goal_config, roadmap, tree, num_neighbors)

    # validate_and_remove_invalid_edges(roadmap1, obstacle_center, half_diagonal, SAFE_ZONE)
    # validate_and_remove_invalid_edges(roadmap2, obstacle_center, half_diagonal, SAFE_ZONE)

    # path_g1, indices_g1 = find_path(roadmap1, start_node_g1, goal_node_g1)

    # path_g2, indices_g2 = find_path(roadmap2, start_node_g2, goal_node_g2)

    # intermediate_joints_g1 = retrieve_joint_configs(indices_g1, roadmap3)
    # intermediate_joints_g2 = retrieve_joint_configs(indices_g2, roadmap3)

    # plot_joint_configs(start_angle, goal_angle, intermediate_joints_g1)
    # plot_joint_configs(start_angle, goal_angle, intermediate_joints_g2)


    # plot_all_joints(joint_angles, roadmap1, roadmap2, roadmap3)

    # compare_edge_distances_histogram(roadmap2, roadmap3)
    
    # compare_edge_distances_scatter(roadmap1,roadmap2, roadmap3)
    
   

    
    
