import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import pickle
import networkx as nx


def load_graph(file_path):
    """
    Load a graph object from a pickle file.

    Args:
    - file_path (str): Path to the pickle file.

    Returns:
    - graph: Loaded graph object.
    """
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    return graph

# Function to visualize a roadmap graph
def visualize_roadmap(graph, title="Roadmap Graph", node_size=20, edge_color='#5D3A9B', node_color='#40B0A6', with_labels=False, layout_seed=42):
    """
    Visualizes a roadmap graph using NetworkX and Matplotlib.

    Args:
    - graph (nx.Graph): The roadmap graph to visualize.
    - title (str): Title of the plot.
    - node_size (int): Size of the nodes in the plot.
    - edge_color (str): Color of the edges in the plot.
    - node_color (str): Color of the nodes in the plot.
    - with_labels (bool): Whether to display labels for the nodes.
    """
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=layout_seed)  # You can use other layouts like `nx.kamada_kawai_layout(graph)`
 



    # Draw nodes and edges
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_color, alpha=0.8)
    nx.draw_networkx_edges(graph, pos, edge_color=edge_color, alpha=0.5)

    if with_labels:
        nx.draw_networkx_labels(graph, pos, font_size=8, font_color='black')

    # plt.title(title)
    plt.show()

# Paths to your stored graphs
custom_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_roadmap_angle_fresh_432.pkl'
euclidean_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_roadmap_angle_fresh_432.pkl'
gt_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_roadmap_angle_fresh_432.pkl'

# Load the graphs
custom_graph = load_graph(custom_graph_path)
euclidean_graph = load_graph(euclidean_graph_path)
gt_graph = load_graph(gt_graph_path)

# Visualize each graph
visualize_roadmap(custom_graph, title="Custom Roadmap Graph")
visualize_roadmap(euclidean_graph, title="Euclidean Roadmap Graph")
visualize_roadmap(gt_graph, title="Ground Truth Roadmap Graph")

