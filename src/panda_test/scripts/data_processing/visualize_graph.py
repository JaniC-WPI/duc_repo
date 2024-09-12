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
import pickle, csv

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

# Load the roadmap and KDTree from files
def load_graph_and_tree(graph_path, tree_path):
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    with open(tree_path, 'rb') as f:
        tree = pickle.load(f)
    print(f"Graph loaded from {graph_path}")
    print(f"KDTree loaded from {tree_path}")
    return graph, tree

def visualize_config(roadmap):
    pos_dict = {n[0]:n[1]["configuration"][8] for n in roadmap.nodes.items()}  
    nx.draw_networkx(roadmap,node_size=5,width=0.3, with_labels=False, pos=pos_dict)
    plt.show()

def plot_joint_angles_3d(graph):
    """
    Plot the joint angles of the nodes in the graph in a 3D scatter plot.

    Args:
    - graph: NetworkX graph where each node contains joint angle information.

    Returns:
    - None
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Extract joint angles for all nodes
    joint_angles = np.array([graph.nodes[node]['joint_angles'] for node in graph.nodes])

    if joint_angles.shape[1] != 3:
        print("Error: Not enough joint angles to plot in 3D.")
        return

    # Plot the joint angles
    ax.scatter(joint_angles[:, 0], joint_angles[:, 1], joint_angles[:, 2], c='orange', marker='_')

    ax.set_xlabel('Joint Angle 1')
    ax.set_ylabel('Joint Angle 2')
    ax.set_zlabel('Joint Angle 3')
    ax.set_title('3D Plot of Joint Angles in the sampled roadmap')

    plt.show()

if __name__ == "__main__":
    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged/'  # Replace with the path to your JSON files
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b128_e500_v32.pth'
    model = load_model_for_inference(model_path)
    custom_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_roadmap_angle.pkl'
    custom_tree_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_tree_angle.pkl'
    
    euclidean_g_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_roadmap_angle.pkl'
    euclidean_tree_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_tree_angle.pkl'

    jt_space_g_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_roadmap_angle.pkl'
    jt_space_tree_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_tree_angle.pkl'

    custom_roadmap, custom_tree = load_graph_and_tree(custom_graph_path, custom_tree_path)

    visualize_config(custom_roadmap)

    euclidean_roadmap, euclidean_tree = load_graph_and_tree(euclidean_g_path, euclidean_tree_path)

    visualize_config(euclidean_roadmap)

    jt_space_roadmap, jt_space_tree = load_graph_and_tree(jt_space_g_path, jt_space_tree_path)

    visualize_config(jt_space_roadmap)

    plot_joint_angles_3d(custom_roadmap)
    plot_joint_angles_3d(euclidean_roadmap)
    plot_joint_angles_3d(jt_space_roadmap)


    
    

 








