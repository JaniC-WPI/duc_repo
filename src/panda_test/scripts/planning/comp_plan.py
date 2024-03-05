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

from pos_regression import PosRegModel

# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
SAFE_DISTANCE = 50  # Safe distance from the obstacle

def load_matched_configurations(directory):
    # Initialize empty lists for configurations
    kp_configurations = []
    jt_configurations = []

    # Temporary dictionary to hold joint angles keyed by identifier
    temp_jt_configurations = {}

    # First pass: Load joint angles into temporary dictionary
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('_joint_angles.json'):
            identifier = filename.replace('_joint_angles.json', '')
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                temp_jt_configurations[identifier] = np.array(data['joint_angles'])

    # Second pass: Match and load keypoints configurations
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.json') and not filename.endswith('_joint_angles.json') and not filename.endswith('_vel.json'):
            identifier = filename.replace('.json', '')
            if identifier in temp_jt_configurations:
                with open(os.path.join(directory, filename), 'r') as file:
                    data = json.load(file)
                    keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]
                    kp_configurations.append(np.array(keypoints))
                    jt_configurations.append(temp_jt_configurations[identifier])

    return kp_configurations, jt_configurations

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
    kp_configurations = kp_configurations[1:9000:500]
    jt_configurations = jt_configurations[1:9000:500]

    flattened_kp_configs = np.vstack([config.flatten() for config in kp_configurations])
    tree1 = KDTree(flattened_kp_configs)
    G1 = nx.Graph() # Graph for KD Tree keypoints configuration
   
    for i, config in enumerate(kp_configurations):
        G1.add_node(i, configuration=config)

    for i, config in enumerate(flattened_kp_configs):
        dist, indices = tree1.query([config], k=k_neighbors + 1)

        for d,j in zip(dist[0],indices[0]):  # Skip self
            if j!=i:
                G1.add_edge(i, j, distance=d)

    tree2 = BallTree(flattened_kp_configs, metric=lambda x, y: predict_custom_distance(x, y, model))
    G2 = nx.Graph() # Graph for Custom Ball Tree keypoints configurations

    for i, config in enumerate(kp_configurations):
        G2.add_node(i, configuration=config)

    for i, config in enumerate(flattened_kp_configs):
        dist, indices = tree2.query([config], k=k_neighbors + 1)  # +1 to include self in results
        #indices = tree.query_radius(config.reshape(1,-1), r=20,count_only=False) # +1 to include self in results
        # print("custom_distance", dist)
        for d,j in zip(dist[0],indices[0]):  # Skip self
            if j !=i:
                G2.add_edge(i, j, distance=d)

    flattened_jt_configs = np.vstack([config.flatten() for config in jt_configurations])
    tree3 = KDTree(flattened_jt_configs)
    G3 = nx.Graph() # Graph for KD Tree joint angles

    for i, config in enumerate(jt_configurations):
        G3.add_node(i, configuration=config)

    for i, config in enumerate(flattened_jt_configs):
        dist, indices = tree3.query([config], k=k_neighbors + 1)
        print("angle distance", dist)
        for d,j in zip(dist[0],indices[0]):  # Skip self
            if j!=i:
                G3.add_edge(i, j, distance=d)    
    # print(G2.nodes.data())
    # print(G3.nodes.data())
    #SG= G.subgraph(range(1,9000,100))
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

def compare_edge_distances(G2, G3):
    """
    Visualize comparison between edge distances of G2 and G3.
    """
    # Gather common edge distances
    common_edge_distances = []
    for (u, v, d) in G2.edges(data=True):
        if G3.has_edge(u, v):
            distance_g2 = d['distance']
            distance_g3 = G3[u][v]['distance']
            common_edge_distances.append((distance_g2, distance_g3))
    
    # Scatter plot for distance comparison
    distances_g2, distances_g3 = zip(*common_edge_distances)
    plt.scatter(distances_g3, distances_g2, alpha=0.5)
    plt.xlabel('Distances in G3')
    plt.ylabel('Distances in G2')
    plt.title('Scatter Plot of Distances in joint_space vs. custom_joint_space')
    plt.grid(True)
    max_distance = max(max(distances_g2), max(distances_g3))
    plt.plot([0, max_distance], [0, max_distance], 'r--')  # y=x reference line
    plt.show()

def compare_edge_distances_histogram(G2, G3):
    """
    Visualize the difference in edge distances between G2 and G3 using a histogram.
    """
    # Gather common edge distances
    common_edge_distances = []
    for (u, v, d) in G2.edges(data=True):
        if G3.has_edge(u, v):
            distance_g2 = d['distance']
            distance_g3 = G3[u][v]['distance']
            # Calculate the difference in distances
            difference = distance_g2 - distance_g3
            common_edge_distances.append(difference)
    
    # Histogram for differences in distance
    plt.hist(common_edge_distances, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Distance Difference (G2 - G3)')
    plt.ylabel('Number of Edges')
    plt.title('Histogram of Distance Differences Between G2 and G3')
    plt.grid(axis='y', alpha=0.75)
    plt.axvline(0, color='red', linestyle='dashed', linewidth=1)  # Line at zero difference
    plt.text(0.1, max(plt.ylim()) * 0.9, 'Equal distances', color = 'red')
    plt.show()

def compare_edge_distance_joint_space(G1,G2,G3):
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

    # Plot histograms
    fig, axs = plt.subplots(3, 3, figsize=(21, 15), sharey=True)
    
    axs[0, 0].hist(distances_g1, bins=30, alpha=0.7, label='G1 Edge Distances', color='blue')
    axs[0, 0].set_title('G1 Edge Distances')

    axs[1, 0].hist(euclidean_distances_g13, bins=30, alpha=0.7, label='G3 Euclidean Distances For Connected nodes', color='blue')
    axs[1, 0].set_title('G3 Euclidean Distances')

    axs[0, 1].hist(distances_g2, bins=30, alpha=0.7, label='G2 Edge Distances', color='blue')
    axs[0, 1].set_title('G2 Edge Distances')

    axs[1, 1].hist(euclidean_distances_g23, bins=30, alpha=0.7, label='G3 Euclidean Distances For Connected nodes', color='blue')
    axs[1, 1].set_title('G3 Euclidean Distances')

    axs[0, 2].hist(distances_g3, bins=30, alpha=0.7, label='G3 Edge Distances', color='blue')
    axs[0, 2].set_title('G3 Edge Distances')

    axs[1, 2].hist(euclidean_distances_g33, bins=30, alpha=0.7, label='G3 Euclidean Distances For Connected nodes', color='blue')
    axs[1, 2].set_title('G3 Euclidean Distances')
            



def compare_all_edge_distances(G1, G2, G3):
    # Initialize lists to store distance differences
    differences_g1_g3 = []
    differences_g2_g3 = []
    
    # Collect distance differences for G1 vs. G3
    for (u, v, d) in G1.edges(data=True):
        
        if G3.has_edge(u,v):            
            distance_g3_g1 = G3[u][v]['distance']
            differences_g1_g3.append(distance_g1 - distance_g3_g1)
    
    # Collect distance differences for G2 vs. G3
    for (u, v, d) in G2.edges(data=True):
        if G3.has_edge(u, v):
            distance_g2 = d['distance']
            distance_g3_g2 = G3[u][v]['distance']
            differences_g2_g3.append(distance_g2 - distance_g3_g2)
    
    # Plot histograms
    fig, axs = plt.subplots(2, 3, figsize=(14, 10), sharey=True)

    axs[0, 0].hist(distances_g1, bins=30, alpha=0.7, label='G1 Edge Distances', color='blue')
    axs[0, 0].set_title('G1 Edge Distances')
    
    axs[0, 1].hist(distances_g2, bins=30, alpha=0.7, label='G2 Edge Distances', color='green')
    axs[0, 1].set_title('G2 Edge Distances')
    
    axs[1, 0].hist(distances_g3, bins=30, alpha=0.7, label='G3 Edge Distances', color='red')
    axs[1, 0].set_title('G3 Edge Distances')
    
    axs[1, 1].hist(euclidean_distances_g3, bins=30, alpha=0.7, label='Euclidean Distances in G3', color='orange')
    axs[1, 1].set_title('Euclidean Distances in G3')
    
    axs[0].hist(differences_g1_g3, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axs[0].set_title('G1 vs. G3')
    axs[0].set_xlabel('Distance Difference (G1 - G3)')
    axs[0].set_ylabel('Number of Edges')
    axs[0].axvline(0, color='red', linestyle='dashed', linewidth=1)
    
    axs[1].hist(differences_g2_g3, bins=30, alpha=0.7, color='green', edgecolor='black')
    axs[1].set_title('G2 vs. G3')
    axs[1].set_xlabel('Distance Difference (G2 - G3)')
    axs[1].axvline(0, color='red', linestyle='dashed', linewidth=1)
    
    # Assuming G3 is your baseline and comparing it to itself just for uniformity in the visualization
    axs[2].hist(differences_g2_g3, bins=30, alpha=0.7, color='orange', edgecolor='black')  # This is a placeholder
    axs[2].set_title('G3 Baseline')
    axs[2].set_xlabel('Distance Difference (G3 - G3)')
    axs[2].axvline(0, color='red', linestyle='dashed', linewidth=1)
    
    plt.tight_layout()
    plt.show()

def calculate_euclidean_distance_3d(pos_u, pos_v):
    """Calculate the Euclidean distance between two 3D positions."""
    return np.linalg.norm(np.array(pos_u) - np.array(pos_v))

def compare_edge_distances(G1, G2, G3):
    # Assuming G3 nodes are already associated with their 3D positions as attributes
    distances_g1 = []
    distances_g2 = []
    distances_g3 = []
    euclidean_distances_g3 = []

    # Process edges for G1 and G2, compare with Euclidean distances in G3
    for G, distances_list in zip([G1, G2, G3], [distances_g1, distances_g2, distances_g3]):
        for (u, v, d) in G.edges(data=True):
            if G3.has_node(u) and G3.has_node(v):
                # Retrieve the 3D positions from G3's nodes
                pos_u = G3.nodes[u]['configuration']
                pos_v = G3.nodes[v]['configuration']
                euclidean_distance = calculate_euclidean_distance_3d(pos_u, pos_v)
                distances_list.append(d['distance'])  # Distance from G1 or G2
                euclidean_distances_g3.append(euclidean_distance)  # Corresponding Euclidean distance in G3

    # Visualization with histograms
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

    axs[0, 0].hist(distances_g1, bins=30, alpha=0.7, label='G1 Edge Distances', color='blue')
    axs[0, 0].set_title('G1 Edge Distances')
    
    axs[0, 1].hist(distances_g2, bins=30, alpha=0.7, label='G2 Edge Distances', color='green')
    axs[0, 1].set_title('G2 Edge Distances')
    
    axs[1, 0].hist(distances_g3, bins=30, alpha=0.7, label='G3 Edge Distances', color='red')
    axs[1, 0].set_title('G3 Edge Distances')
    
    axs[1, 1].hist(euclidean_distances_g3, bins=30, alpha=0.7, label='Euclidean Distances in G3', color='orange')
    axs[1, 1].set_title('Euclidean Distances in G3')
    
    for ax in axs.flat:
        ax.label_outer()  # Hide x labels and tick labels for top plots and y ticks for right plots.
        ax.legend()
    
    plt.show()

    # Visualization with histograms
    plt.hist(euclidean_distances_g3, bins=30, alpha=0.5, label='Euclidean Distances in G3', color='orange', edgecolor='black')
    # plt.hist(distances_g1, bins=30, alpha=0.5, label='Edge Distances in G1', color='blue', edgecolor='black')
    # plt.hist(distances_g2, bins=30, alpha=0.5, label='Edge Distances in G2', color='green', edgecolor='black')
    plt.hist(distances_g3, bins=30, alpha=0.5, label='Edge Distances in G3', color='brown', edgecolor='black')
    plt.xlabel('Distances')
    plt.ylabel('Number of Edges')
    plt.title('Comparison of Edge Distances: G1 & G2 & G3 vs. Euclidean in G3')
    plt.legend()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_planning_kprcnn/' 
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b64_e400_v6.pth'    
    kp_configurations, joint_angles = load_matched_configurations(directory)
    model = load_model_for_inference(model_path)    
    num_neighbors = 10 # Number of neighbors for each node in the roadmap
    # Build the roadmaps
    roadmap1, roadmap2, roadmap3, tree1, tree2, tree3 = build_lazy_roadmap(kp_configurations, joint_angles, num_neighbors, model)   

    # compare_edge_distances_histogram(roadmap2, roadmap3)
    compare_edge_distances(roadmap1,roadmap2,roadmap3)

    
    
