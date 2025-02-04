import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import pickle
import networkx as nx
from sklearn.manifold import TSNE

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
    # plt.show()

# Paths to your stored graphs
custom_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_roadmap_angle_fresh_all_432_edges_dist_check.pkl'
euclidean_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_roadmap_angle_fresh_all_432_edges_dist_check.pkl'
gt_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_roadmap_angle_fresh_all_432_edges_dist_check.pkl'

# Load the graphs
custom_graph = load_graph(custom_graph_path)
euclidean_graph = load_graph(euclidean_graph_path)
gt_graph = load_graph(gt_graph_path)

# Visualize each graph
visualize_roadmap(custom_graph, title="Custom Roadmap Graph")
visualize_roadmap(euclidean_graph, title="Euclidean Roadmap Graph")
visualize_roadmap(gt_graph, title="Ground Truth Roadmap Graph")

# Save edge list
nx.write_edgelist(custom_graph, "/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/edge_details/custom_graph_edgelist.txt", data=True)

# Save adjacency matrix
import numpy as np
adj_matrix = nx.adjacency_matrix(custom_graph).todense()
np.savetxt("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/edge_details/custom_graph_adjacency_matrix.txt", adj_matrix)

# Save edge list
nx.write_edgelist(custom_graph, "/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/edge_details/euclidean_graph_edgelist.txt", data=True)

# Save adjacency matrix
import numpy as np
adj_matrix = nx.adjacency_matrix(custom_graph).todense()
np.savetxt("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/edge_details/euclidean_graph_adjacency_matrix.txt", adj_matrix)

# Save edge weights
import csv
with open("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/edge_details/custom_graph_edge_weights.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Source", "Target", "Weight"])
    for u, v, d in custom_graph.edges(data=True):
        writer.writerow([u, v, d.get("weight", 1)])  # Default weight = 1

with open("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/edge_details/euclidean_graph_edge_weights.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Source", "Target", "Weight"])
    for u, v, d in euclidean_graph.edges(data=True):
        writer.writerow([u, v, d.get("weight", 1)])  # Default weight = 1

custom_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_roadmap_angle_fresh_432.pkl'
euclidean_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_roadmap_angle_fresh_432.pkl'
gt_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_roadmap_angle_fresh_432.pkl'

# Load the graphs
custom_graph = load_graph(custom_graph_path)
euclidean_graph = load_graph(euclidean_graph_path)
gt_graph = load_graph(gt_graph_path)

# Load the graphs from .pkl files
with open(custom_graph_path, "rb") as f:
    custom_graph = pickle.load(f)

with open(euclidean_graph_path, "rb") as f:
    euclidean_graph = pickle.load(f)

    # Check if 'pos' exists in the graph nodes
has_pos = all("pos" in data for _, data in custom_graph.nodes(data=True))
print(f"Do all nodes have a 'pos' attribute? {'Yes' if has_pos else 'No'}")

# Display a sample of node attributes
for node, data in list(custom_graph.nodes(data=True))[:5]:  # Show only the first 5 nodes
    print(f"Node: {node}, Data: {data}")

# Extract node features using the "configuration" attribute
def get_node_features(graph):
    return np.array([data["configuration"].flatten() for _, data in graph.nodes(data=True)])

custom_features = get_node_features(custom_graph)
euclidean_features = get_node_features(euclidean_graph)
gt_features = get_node_features(gt_graph)

# Perform t-SNE for dimensionality reduction
tsne_custom = TSNE(n_components=2, random_state=42).fit_transform(custom_features)
tsne_euclidean = TSNE(n_components=2, random_state=42).fit_transform(euclidean_features)
tsne_gt = TSNE(n_components=2, random_state=42).fit_transform(gt_features)


# Visualize the t-SNE embeddings
plt.figure(figsize=(12, 6))

# Custom graph t-SNE
plt.subplot(1, 2, 1)
plt.scatter(tsne_custom[:, 0], tsne_custom[:, 1], alpha=0.7, s=10, c='blue')
plt.title("t-SNE Visualization of Learned Roadmap")
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
plt.grid(True)

# Euclidean graph t-SNE
plt.subplot(1, 2, 2)
plt.scatter(tsne_euclidean[:, 0], tsne_euclidean[:, 1], alpha=0.7, s=10, c='orange')
plt.title("t-SNE Visualization of Image Space Roadmap")
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
plt.grid(True)

plt.tight_layout()
# plt.show()

# Helper function to plot t-SNE with edges
def plot_tsne_with_edges(tsne_data, graph, method, color):
    plt.figure(figsize=(8, 8))
    # Plot nodes
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], alpha=0.7, s=100, c=color, label="Nodes")

    # Plot edges
    for edge in graph.edges():
        node1, node2 = edge
        x_coords = [tsne_data[node1, 0], tsne_data[node2, 0]]
        y_coords = [tsne_data[node1, 1], tsne_data[node2, 1]]
        plt.plot(x_coords, y_coords, c="gray", alpha=0.5, linewidth=0.5)

    plt.title("")
    plt.xticks(fontsize=22)  # Increase x-axis tick label size
    plt.yticks(fontsize=22)
    # plt.xlabel("t-SNE Dimension 1")
    # plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True)
    # plt.savefig(f'/media/jc-merlab/Crucial X9/paper_data/tsne_{method}.png')
    plt.show()

# Visualize t-SNE with edges
# plot_tsne_with_edges(tsne_custom, custom_graph, "t-SNE with Edges: Learned Roadmap", color="blue")
# plot_tsne_with_edges(tsne_euclidean, euclidean_graph, "t-SNE with Edges: Image Space Roadmap", color="orange")

plot_tsne_with_edges(tsne_custom, custom_graph, "learned", color="blue")
plot_tsne_with_edges(tsne_euclidean, euclidean_graph, "image", color="orange")
plot_tsne_with_edges(tsne_gt, gt_graph, "gt", color="black")



