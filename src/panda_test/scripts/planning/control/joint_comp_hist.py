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

# Function to plot joint angles in 3D for a single roadmap
def plot_joint_angles_3d(roadmap, label, output_path):
    joint_angles = [roadmap.nodes[node]['joint_angles'] for node in roadmap.nodes]

    joint_1 = [angles[0] for angles in joint_angles]
    joint_2 = [angles[1] for angles in joint_angles]
    joint_3 = [angles[2] for angles in joint_angles]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the joint angles
    ax.scatter(joint_1, joint_2, joint_3, marker='o', label=label)

    ax.set_xlabel('Joint 1')
    ax.set_ylabel('Joint 2')
    ax.set_zlabel('Joint 3')

    plt.title(f'3D Joint Angle Plot for {label}')
    plt.legend()
    plt.savefig(output_path)
    plt.show()
    plt.close()

# Function to plot joint angles in 3D for all roadmaps together
def plot_joint_angles_3d_all(roadmaps, labels, output_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for roadmap, label in zip(roadmaps, labels):
        joint_angles = [roadmap.nodes[node]['joint_angles'] for node in roadmap.nodes]

        joint_1 = [angles[0] for angles in joint_angles]
        joint_2 = [angles[1] for angles in joint_angles]
        joint_3 = [angles[2] for angles in joint_angles]

        # Plot joint angles for this roadmap
        ax.scatter(joint_1, joint_2, joint_3, marker='o', label=label)

    ax.set_xlabel('Joint 1')
    ax.set_ylabel('Joint 2')
    ax.set_zlabel('Joint 3')

    plt.title('3D Joint Angle Plot for All Roadmaps')
    ax.legend()
    plt.savefig(output_path)
    plt.show()
    plt.close()

def plot_connected_joints_3d(roadmap, color, label, output_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Loop over edges to get the connected nodes
    for edge in roadmap.edges():
        node1, node2 = edge

        joint_angles_1 = roadmap.nodes[node1]['joint_angles']
        joint_angles_2 = roadmap.nodes[node2]['joint_angles']

        joint_1 = [joint_angles_1[0], joint_angles_2[0]]
        joint_2 = [joint_angles_1[1], joint_angles_2[1]]
        joint_3 = [joint_angles_1[2], joint_angles_2[2]]

        # Plot the line between connected nodes (configurations)
        ax.plot(joint_1, joint_2, joint_3, marker='o', color=color)

    ax.set_xlabel('Joint 1')
    ax.set_ylabel('Joint 2')
    ax.set_zlabel('Joint 3')

    plt.title(f'Connected Joints 3D Plot for {label}')
    plt.savefig(output_path)
    plt.show()
    plt.close()

# Function to plot connected joints in 3D for all roadmaps together
def plot_connected_joints_3d_all(roadmaps, labels, output_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['b', 'g', 'r']  # Different colors for different roadmaps

    for roadmap, label, color in zip(roadmaps, labels, colors):
        # Loop over edges to get the connected nodes
        for edge in roadmap.edges():
            node1, node2 = edge

            joint_angles_1 = roadmap.nodes[node1]['joint_angles']
            joint_angles_2 = roadmap.nodes[node2]['joint_angles']

            joint_1 = [joint_angles_1[0], joint_angles_2[0]]
            joint_2 = [joint_angles_1[1], joint_angles_2[1]]
            joint_3 = [joint_angles_1[2], joint_angles_2[2]]

            # Plot the line between connected nodes (configurations) for this roadmap
            ax.plot(joint_1, joint_2, joint_3, marker='o', label=label if edge == list(roadmap.edges())[0] else "", color=color)

    ax.set_xlabel('Joint 1')
    ax.set_ylabel('Joint 2')
    ax.set_zlabel('Joint 3')

    plt.title('Connected Joints 3D Plot for All Roadmaps')
    ax.legend()
    plt.savefig(output_path)
    plt.show()
    plt.close()

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
    Plot histograms of joint angle distances for multiple datasets with same x and y scaling.

    Args:
    - distance_list (list of lists): List of joint angle distances for each dataset.
    - labels (list of str): List of labels for the datasets.
    - output_prefix (str): Prefix for saving the output plot files.
    - fontsize (int): Font size for plot labels and titles.

    Returns:
    - None
    """
    plt.figure(figsize=(15, 5))  # Create a figure for the histograms

    # Determine the common x-axis and y-axis limits
    max_distance = max(max(distances) for distances in distance_list)
    max_count = max(np.histogram(distances, bins=30)[0].max() for distances in distance_list)

    # Plot histograms for the datasets
    for i, distances in enumerate(distance_list):
        plt.subplot(1, 3, i + 1)
        plt.hist(distances, bins=10, alpha=0.3, color=colors[i], label=labels[i])
        plt.title(f'{labels[i]} - Joint Angle Distances', fontsize=fontsize + 2)
        plt.xlabel('Joint Angle Distance', fontsize=fontsize)
        plt.ylabel('Number of Connected Edges', fontsize=fontsize)
        plt.legend(fontsize=fontsize)

        # Set the same x-axis and y-axis limits for all histograms
        plt.xlim(0, 2.7)
        # plt.ylim(0, (max_count+1000))

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

    # plot_joint_angles_3d(custom_graph, 'Custom Distance', 'custom_roadmap_joint_angles.png')
    # plot_joint_angles_3d(euclidean_graph, 'Euclidean Distance', 'euclidean_roadmap_joint_angles.png')
    # plot_joint_angles_3d(joint_space_graph, 'Ground Truth Distance', 'gt_roadmap_joint_angles.png')

    # # Plot for all roadmaps together
    # roadmaps = [custom_graph, euclidean_graph, joint_space_graph]
    # labels = ['Custom Distance', 'Euclidean Distance', 'Ground Truth Distance']
    # plot_joint_angles_3d_all(roadmaps, labels, 'all_roadmaps_joint_angles.png')

    # plot_connected_joints_3d(custom_graph, 'b', 'Custom Distance', '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_roadmap_n3_joint_angles.png')
    # plot_connected_joints_3d(euclidean_graph, 'r', 'Euclidean Distance', '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_roadmap_n3_joint_angles.png')
    # plot_connected_joints_3d(joint_space_graph, 'g', 'Ground Truth Distance', '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/gt_roadmap_n3_joint_angles.png')

    # # Plot for all roadmaps together
    # roadmaps = [custom_graph, euclidean_graph, joint_space_graph]
    # labels = ['Custom Distance', 'Euclidean Distance', 'Ground Truth Distance']
    # plot_connected_joints_3d_all(roadmaps, labels, 'all_roadmaps_joint_angles.png')


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
