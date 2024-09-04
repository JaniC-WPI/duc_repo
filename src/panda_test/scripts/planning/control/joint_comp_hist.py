import pickle
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from itertools import islice

import pickle
import matplotlib.pyplot as plt
import numpy as np
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

def calculate_joint_angle_distances(graph):
    """
    Calculate joint angle distances for all edges in a graph.

    Args:
    - graph: Graph object containing nodes with joint angle data.

    Returns:
    - distances (list of float): List of joint angle distances between nodes connected by edges.
    """
    distances = []
    for u, v in graph.edges():
        if 'joint_angles' in graph.nodes[u] and 'joint_angles' in graph.nodes[v]:
            joint_angles_u = graph.nodes[u]['joint_angles']
            joint_angles_v = graph.nodes[v]['joint_angles']
            distance = np.linalg.norm(joint_angles_u - joint_angles_v)
            distances.append(distance)

    return distances

def find_common_edges(graph1, graph2):
    """
    Find edges that are common between two graphs.

    Args:
    - graph1: First graph object.
    - graph2: Second graph object.

    Returns:
    - common_edges (set of tuples): Set of edges common to both graphs.
    """
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())
    common_edges = edges1.intersection(edges2)
    return len(common_edges)

def print_graph_statistics(graph, label):
    """
    Print detailed statistics about the graph for debugging.

    Args:
    - graph: Graph object.
    - label: Label for the graph being analyzed.

    Returns:
    - None
    """
    print(f"\n{label} - Graph Statistics:")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    
    # Print some sample edges and their weights
    edges_sample = list(islice(graph.edges(data=True), 5))  # Print first 5 edges
    print("Sample edges and weights:")
    for u, v, data in edges_sample:
        print(f"Edge ({u}, {v}) - Weight: {data.get('weight', 'N/A')}")

def plot_joint_distance_histograms(distance_list, labels, colors, output_prefix, fontsize=12):
    """
    Plot histograms of joint angle distances for multiple datasets.

    Args:
    - distance_list (list of lists): List of joint angle distances for each dataset.
    - labels (list of str): List of labels for the datasets.
    - output_prefix (str): Prefix for saving the output plot files.

    Returns:
    - None
    """
    plt.figure(figsize=(15, 5))  # Create a figure for the histograms

    # Plot histograms for the datasets
    for i, distances in enumerate(distance_list):
        plt.subplot(1, 3, i + 1)
        plt.hist(distances, bins=30, alpha=0.3, color=colors[i], label=labels[i])
        plt.title(f'{labels[i]} - Joint Angle Distances', fontsize=fontsize + 2)
        plt.xlabel('Joint Angle Distance', fontsize=fontsize)
        plt.ylabel('Joint Angles in Edges', fontsize=fontsize)
        plt.legend(fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_joint_angle_distances.png')
    plt.show()

def main():
    # Paths to the saved roadmap files
    custom_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_roadmap_angle.pkl'
    euclidean_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_roadmap_angle.pkl'
    joint_space_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_roadmap_angle.pkl'

    # Load the roadmaps
    custom_graph = load_graph(custom_graph_path)
    euclidean_graph = load_graph(euclidean_graph_path)
    joint_space_graph = load_graph(joint_space_graph_path)

    common_edge_custom = find_common_edges(custom_graph, joint_space_graph)
    common_edge_euclidean = find_common_edges(euclidean_graph, joint_space_graph)

    print(common_edge_custom, common_edge_euclidean)

    # Print graph statistics
    print_graph_statistics(custom_graph, "Custom PRM")
    print_graph_statistics(euclidean_graph, "Euclidean PRM")
    print_graph_statistics(joint_space_graph, "Joint Space PRM")

    # Calculate joint angle distances for edges in each roadmap
    custom_joint_distances = calculate_joint_angle_distances(custom_graph)
    euclidean_joint_distances = calculate_joint_angle_distances(euclidean_graph)
    joint_space_joint_distances = calculate_joint_angle_distances(joint_space_graph)

    # Ensure that distance lists are not empty
    if len(custom_joint_distances) == 0 or len(euclidean_joint_distances) == 0 or len(joint_space_joint_distances) == 0:
        print("Error: One or more joint angle distance datasets are empty. Please check the input data.")
        return

    # Plot histograms for joint angle distances
    distance_list = [custom_joint_distances, euclidean_joint_distances, joint_space_joint_distances]
    labels = ['Custom PRM', 'Euclidean PRM', 'Joint Space PRM']
    colors = ['blue', 'green', 'orange']
    output_prefix = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_angle_distances'
    plot_joint_distance_histograms(distance_list, labels, colors, output_prefix)

if __name__ == "__main__":
    main()
