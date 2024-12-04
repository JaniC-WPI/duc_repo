# import json
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# import pandas as pd
# import pickle


# def load_graph(file_path):
#     """
#     Load a graph object from a pickle file.

#     Args:
#     - file_path (str): Path to the pickle file.

#     Returns:
#     - graph: Loaded graph object.
#     """
#     with open(file_path, 'rb') as f:
#         graph = pickle.load(f)
#     return graph

# def plot_joint_angle_histograms(roadmap, label, output_path):
#     # Extract joint angles from the roadmap
#     joint_angles = [roadmap.nodes[node]['joint_angles'] for node in roadmap.nodes]
#     joint_1 = [angles[0] for angles in joint_angles]
#     joint_2 = [angles[1] for angles in joint_angles]
#     joint_3 = [angles[2] for angles in joint_angles]

#     # Create a DataFrame for Seaborn visualizations
#     joint_angles_df = pd.DataFrame({
#         'Joint 1': joint_1,
#         'Joint 2': joint_2,
#         'Joint 3': joint_3
#     })

#     # Calculate the common range and set a consistent bin width
#     min_value = min(joint_angles_df.min())
#     max_value = max(joint_angles_df.max())
#     bin_width = 0.25  # Set your desired bin width
#     bins = np.arange(min_value, max_value + bin_width, bin_width)  # Create bins with the same width

#     # Plot individual histograms for each joint angle with the same bin width
#     plt.figure(figsize=(20, 10))
#     for i, joint in enumerate(joint_angles_df.columns, 1):
#         plt.subplot(1, 3, i)
#         sns.histplot(joint_angles_df[joint], bins=bins, kde=False, color='#5D3A9B', alpha = 0.9)
#         # plt.title(f'Distribution of {joint}', fontsize=14, fontweight='bold')
#         # plt.xlabel(joint, fontsize=12, fontweight='bold')
#         # plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        
#         plt.xlabel('')
#         plt.ylabel('')
#         plt.xlim(-3, 3)
#         plt.ylim(0, 5301)
#         plt.tick_params(axis='y', pad=5)
#         plt.xticks(fontsize=32, weight='bold')
#         # plt.yticks(fontsize=32, weight='bold')
#         if i == 1:
#             # plt.yticks(fontsize=32, weight='bold')
#             plt.yticks(np.arange(0, 5301, 500), fontsize=32, weight='bold')
#         else:
#             plt.yticks([])
        


#     # plt.suptitle(f'Joint Angle Distributions for {label}', fontsize=16, fontweight='bold', y=1.02)
#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.show()
#     plt.close()

# folder_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged_og/'

# custom_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_roadmap_angle_fresh_432.pkl'
# gt_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_roadmap_angle_fresh_432.pkl'

# gt_graph = load_graph(gt_graph_path)


# # Initialize lists to hold joint angles
# j1_angles = []
# j2_angles = []
# j3_angles = []

# # Iterate over each file in the folder
# for filename in sorted(os.listdir(folder_path)):
#     if filename.endswith('_joint_angles.json'):  # Ensures we only read .json files
#         # Check if the file index is greater than or equal to 010000
#         file_index = int(filename.split('_')[0])
#         # if file_index >= 10000:
#         file_path = os.path.join(folder_path, filename)
        
#         # Open and read the JSON file
#         with open(file_path, 'r') as file:
#             data = json.load(file)
#             joint_angles = data['joint_angles']
            
#             # Append each angle to its respective list
#             j1_angles.append(joint_angles[0])
#             j2_angles.append(joint_angles[1])
#             j3_angles.append(joint_angles[2])

# # Create a DataFrame for Seaborn visualizations
# joint_angles_df = pd.DataFrame({
#     'Joint 1': j1_angles,
#     'Joint 2': j2_angles,
#     'Joint 3': j3_angles
# })

# # 3D Scatter Plot using Matplotlib
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Calculate the common range and set a consistent bin width
# min_value = min(joint_angles_df.min())
# max_value = max(joint_angles_df.max())
# bin_width = 0.25  # Set your desired bin width
# bins = np.arange(min_value, max_value + bin_width, bin_width)

# plt.figure(figsize=(20, 10))
# # sns.set_theme(style='white')
# for i, joint in enumerate(joint_angles_df.columns, 1):
#     plt.subplot(1, 3, i)    
#     sns.histplot(joint_angles_df[joint], bins=bins, kde=False, color='#517119', edgecolor="black",
#             alpha=0.9)
#     # plt.title(f'Distribution of {joint}', fontsize=14, fontweight='bold')
#     # plt.xlabel(joint, fontsize=12, fontweight='bold')
#     # plt.ylabel('Frequency', fontsize=12, fontweight='bold')
#     plt.xlabel('')
#     plt.ylabel('')
#     # plt.grid(False)
#     plt.xlim(-3, 3)
#     # plt.ylim(-1, 5300)
#     plt.xticks(fontsize=32, weight='bold')
#     # plt.yticks(fontsize=32, weight='bold')
#     if i == 1:
#             # plt.yticks(fontsize=30, weight='bold')
#             plt.yticks(np.arange(0, 5301, 500), fontsize=32, weight='bold')        
#     else:
#         plt.yticks([])

# plt.tight_layout()
# plt.show()
# plt.close()

# plot_joint_angle_histograms(gt_graph, 'Custom Distance', 'custom_roadmap_joint_angles.png')

# import json
# import os
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pickle

# def load_graph(file_path):
#     """
#     Load a graph object from a pickle file.

#     Args:
#     - file_path (str): Path to the pickle file.

#     Returns:
#     - graph: Loaded graph object.
#     """
#     with open(file_path, 'rb') as f:
#         graph = pickle.load(f)
#     return graph

# def scale_frequencies(data, scale_factor):
#     """
#     Scale the frequencies of a histogram.

#     Args:
#     - data (pd.DataFrame): Data for the histogram.
#     - scale_factor (float): Factor to scale the frequencies.

#     Returns:
#     - scaled_data (pd.DataFrame): Scaled histogram data.
#     """
#     return data * scale_factor

# def plot_histograms_with_normalized_frequency(data1, data2, label1, label2, bins, output_path):
#     """
#     Plot histograms with normalized frequencies.

#     Args:
#     - data1 (pd.DataFrame): Data for the larger dataset.
#     - data2 (pd.DataFrame): Data for the smaller dataset.
#     - label1 (str): Label for dataset 1.
#     - label2 (str): Label for dataset 2.
#     - bins (array): Bin edges for histograms.
#     - output_path (str): Path to save the plot.
#     """
#     plt.figure(figsize=(20, 10))
#     for i, joint in enumerate(data1.columns, 1):
#         plt.subplot(1, 3, i)
#         # Plot original dataset
#         sns.histplot(data1[joint], bins=bins, kde=False, color='#517119', alpha=0.6, label=label1)

#         # Plot scaled subset dataset
#         sns.histplot(data2[joint], bins=bins, kde=False, color='#EF4026', alpha=0.6, label=label2)

#         # Axis labels and limits
#         plt.xlabel('')
#         plt.ylabel('')
#         plt.xlim(-3, 3)
#         plt.xticks(fontsize=32, weight='bold')
#         plt.ylim(0, max_y)  # Keep consistent y-scale

#         if i == 1:
#             plt.yticks(np.arange(0, max_y + 1, max_y // 10), fontsize=32, weight='bold')
#         else:
#             plt.yticks([])

#         plt.legend(fontsize=20, loc='upper right', frameon=False)

#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.show()
#     plt.close()

# # Load Dataset 1 (Folder dataset)
# folder_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged_og/'
# j1_angles = []
# j2_angles = []
# j3_angles = []

# # Extract joint angles from JSON files
# for filename in sorted(os.listdir(folder_path)):
#     if filename.endswith('_joint_angles.json'):
#         file_path = os.path.join(folder_path, filename)
#         with open(file_path, 'r') as file:
#             data = json.load(file)
#             joint_angles = data['joint_angles']
#             j1_angles.append(joint_angles[0])
#             j2_angles.append(joint_angles[1])
#             j3_angles.append(joint_angles[2])

# # Create DataFrame for joint angles
# joint_angles_df = pd.DataFrame({
#     'Joint 1': j1_angles,
#     'Joint 2': j2_angles,
#     'Joint 3': j3_angles
# })

# # Load Dataset 2 (Graph data)
# gt_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_roadmap_angle_fresh_432.pkl'
# gt_graph = load_graph(gt_graph_path)

# gt_joint_angles = pd.DataFrame({
#     'Joint 1': [gt_graph.nodes[node]['joint_angles'][0] for node in gt_graph.nodes],
#     'Joint 2': [gt_graph.nodes[node]['joint_angles'][1] for node in gt_graph.nodes],
#     'Joint 3': [gt_graph.nodes[node]['joint_angles'][2] for node in gt_graph.nodes]
# })

# # Calculate scale factor for frequency normalization
# scale_factor = len(joint_angles_df) / len(gt_joint_angles)

# # Scale frequencies of Dataset 2
# scaled_gt_joint_angles = gt_joint_angles.copy()

# # Define bins for histograms
# bin_width = 0.25
# bins = np.arange(-3, 3 + bin_width, bin_width)

# # Determine maximum y-axis range based on Dataset 1
# max_y = joint_angles_df.apply(lambda col: np.histogram(col, bins=bins)[0].max()).max()

# # Plot histograms
# plot_histograms_with_normalized_frequency(joint_angles_df, scaled_gt_joint_angles, 
#                                           'Original Dataset', 'Scaled Subset Dataset', 
#                                           bins, 'joint_angles_normalized_frequency.png')


import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def load_graph(file_path):
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    return graph

def scale_frequencies(data, scale_factor):
    """Scale histogram frequencies."""
    return data * scale_factor

def plot_separate_histograms(data1, data2, label1, label2, bins, output_path):
    """
    Plot separate histograms with consistent y-axis scaling.
    
    Args:
    - data1: Larger dataset (DataFrame).
    - data2: Smaller dataset (DataFrame).
    - label1: Label for larger dataset.
    - label2: Label for smaller dataset.
    - bins: Bins for the histograms.
    - output_path: Path to save the plot.
    """
    plt.figure(figsize=(20, 15))
    for i, joint in enumerate(data1.columns, 1):
        # Create histograms
        hist1, _ = np.histogram(data1[joint], bins=bins)
        hist2, _ = np.histogram(data2[joint], bins=bins)
        
        # Scale smaller dataset frequencies
        scale_factor = len(data1) / len(data2)
        scaled_hist2 = hist2 * scale_factor

        print(hist1.max(), hist2.max())
        
        # Plot histograms separately
        plt.subplot(1, 3, i)  # Larger dataset
        sns.histplot(x=bins[:-1], y=hist1, color="#517119", alpha=0.9, label=label1, edgecolor='black')
        plt.ylim(0, max(hist1.max(), scaled_hist2.max())+500)
        # plt.xticks(fontsize=14, weight="bold")
        plt.xticks([])
        plt.yticks(fontsize=14, weight="bold")
        # plt.title(f"{joint} ({label1})", fontsize=16, weight="bold")
        
        plt.subplot(1, 3, i)  # Smaller dataset
        sns.histplot(x=bins[:-1], y=scaled_hist2, color="#5D3A9B", alpha=0.9, label=label2, edgecolor='black')
        plt.ylim(0, max(hist1.max(), scaled_hist2.max())+500)
        # x_ticks = np.arange(-3, 4, 0.5)  # Increase x-tick steps
        plt.xticks(np.arange(bins.min(), bins.max() + 0.5, 1), fontsize=14, weight="bold") 
        # plt.xticks(x_ticks, fontsize=14, weight="bold")
        plt.yticks(fontsize=14, weight="bold")
        # plt.title(f"{joint} ({label2}, Scaled)", fontsize=16, weight="bold")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# Load larger dataset (folder dataset)
folder_path = "/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged_og/"
j1_angles, j2_angles, j3_angles = [], [], []

for filename in sorted(os.listdir(folder_path)):
    if filename.endswith("_joint_angles.json"):
        with open(os.path.join(folder_path, filename), "r") as file:
            data = json.load(file)
            joint_angles = data["joint_angles"]
            j1_angles.append(joint_angles[0])
            j2_angles.append(joint_angles[1])
            j3_angles.append(joint_angles[2])

joint_angles_df = pd.DataFrame({
    "Joint 1": j1_angles,
    "Joint 2": j2_angles,
    "Joint 3": j3_angles,
})

# Load smaller dataset (graph data)
gt_graph_path = "/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_roadmap_angle_fresh_432.pkl"
gt_graph = load_graph(gt_graph_path)
gt_joint_angles = pd.DataFrame({
    "Joint 1": [gt_graph.nodes[node]["joint_angles"][0] for node in gt_graph.nodes],
    "Joint 2": [gt_graph.nodes[node]["joint_angles"][1] for node in gt_graph.nodes],
    "Joint 3": [gt_graph.nodes[node]["joint_angles"][2] for node in gt_graph.nodes],
})

# Define bins
bin_width = 0.25
bins = np.arange(-3, 3 + bin_width, bin_width)

# Plot histograms
plot_separate_histograms(joint_angles_df, gt_joint_angles, 
                         "Original Dataset", "Subset Dataset", bins, 
                         "normalized_joint_angle_histograms.png")

def load_graph(file_path):
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    return graph


def plot_histograms_with_scaling(data1, data2, label1, label2, bins, output_path1, output_path2):
    """
    Plot histograms for two datasets in separate figures with scaling applied to the second dataset.
    
    Args:
    - data1: Larger dataset (DataFrame).
    - data2: Smaller dataset (DataFrame).
    - label1: Label for larger dataset.
    - label2: Label for smaller dataset.
    - bins: Bins for the histograms.
    - output_path1: Path to save the plot for the first dataset.
    - output_path2: Path to save the plot for the second dataset.
    """
    # Calculate scale factor for smaller dataset
    scale_factor = len(data1) / len(data2)

    # Create figure for dataset 1 (original dataset)
    plt.figure(figsize=(20, 10))
    for i, joint in enumerate(data1.columns, 1):
        plt.subplot(1, 3, i)
        sns.histplot(data=data1, x=joint, bins=bins, color="#517119", alpha=0.9, kde=False, edgecolor="black")
        plt.ylim(0, 5500)
        plt.xticks(np.arange(bins.min(), bins.max() + 1, 1), fontsize=32, weight="bold")
        if i == 1:  # Show y-ticks only for the first joint
            plt.yticks(fontsize=32, weight="bold")
            plt.ylabel('')
        else:
            plt.yticks([])  # Remove y-ticks for other subplots
            plt.ylabel('')
        # plt.xlabel("Joint positions (radians)", fontsize=14, weight="bold")
        plt.xlabel('')
        # plt.ylabel("Configuration space", fontsize=14, weight="bold")
        # plt.title(f"{joint} ({label1})", fontsize=16, weight="bold")
        plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_path1)
    plt.show()

    # Create figure for dataset 2 (scaled subset dataset)
    plt.figure(figsize=(20, 10))
    for i, joint in enumerate(data2.columns, 1):
        plt.subplot(1, 3, i)

        # Compute the histogram manually for data2 and scale it
        hist, edges = np.histogram(data2[joint], bins=bins)
        scaled_hist = hist * scale_factor

        # Use sns.histplot with precomputed bins and frequencies
        sns.histplot(
            x=edges[:-1], weights=scaled_hist, bins=bins, color="#5D3A9B", alpha=0.9, edgecolor="black", kde=False
        )

        plt.ylim(0, 5500)  # Match y-scale to dataset 1
        plt.xticks(np.arange(bins.min(), bins.max() + 1, 1), fontsize=32, weight="bold")
        if i == 1:  # Show y-ticks only for the first joint
            plt.yticks(fontsize=32, weight="bold")
            plt.ylabel('')
        else:
            plt.yticks([])  # Remove y-ticks for other subplots
            plt.ylabel('')
        # plt.xlabel("Joint positions (radians)", fontsize=14, weight="bold")
        plt.xlabel('')            
        # plt.ylabel("Configuration subset", fontsize=14, weight="bold")
        # plt.title(f"{joint} ({label2}, Scaled)", fontsize=16, weight="bold")
        # plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_path2)
    plt.show()


# Load larger dataset (folder dataset)
folder_path = "/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged_og/"
j1_angles, j2_angles, j3_angles = [], [], []

for filename in sorted(os.listdir(folder_path)):
    if filename.endswith("_joint_angles.json"):
        with open(os.path.join(folder_path, filename), "r") as file:
            data = json.load(file)
            joint_angles = data["joint_angles"]
            j1_angles.append(joint_angles[0])
            j2_angles.append(joint_angles[1])
            j3_angles.append(joint_angles[2])

joint_angles_df = pd.DataFrame({
    "Joint 1": j1_angles,
    "Joint 2": j2_angles,
    "Joint 3": j3_angles,
})

# Load smaller dataset (graph data)
gt_graph_path = "/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_roadmap_angle_fresh_432.pkl"
gt_graph = load_graph(gt_graph_path)
gt_joint_angles = pd.DataFrame({
    "Joint 1": [gt_graph.nodes[node]["joint_angles"][0] for node in gt_graph.nodes],
    "Joint 2": [gt_graph.nodes[node]["joint_angles"][1] for node in gt_graph.nodes],
    "Joint 3": [gt_graph.nodes[node]["joint_angles"][2] for node in gt_graph.nodes],
})

# Define bins
bin_width = 0.25
bins = np.arange(-3, 3 + bin_width, bin_width)

# Plot histograms
# plot_histograms_with_scaling(
#     joint_angles_df,
#     gt_joint_angles,
#     "Original Dataset",
#     "Subset Dataset",
#     bins,
#     "/media/jc-merlab/Crucial X9/paper_data/og_ds_hist.png",
#     "/media/jc-merlab/Crucial X9/paper_data/rm_ds_hist.png",
# )

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Example data
nodes = np.random.rand(2462, 9, 2)  # Replace with your actual data
flattened_nodes = nodes.reshape(2462, -1)  # Shape: (2462, 18)

# Apply PCA
pca = PCA(n_components=2)
nodes_2d = pca.fit_transform(flattened_nodes)

# Plot the 2D projection
plt.scatter(nodes_2d[:, 0], nodes_2d[:, 1], s=5, alpha=0.7)
plt.title("Roadmap Visualization in 2D (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid()
plt.show()

# from sklearn.manifold import TSNE

# tsne = TSNE(n_components=2, random_state=42)
# nodes_tsne = tsne.fit_transform(flattened_nodes)

# plt.scatter(nodes_tsne[:, 0], nodes_tsne[:, 1], s=5, alpha=0.7, c=range(len(nodes_tsne)), cmap='viridis')
# plt.title("Roadmap Visualization in 2D (t-SNE Projection)")
# plt.colorbar(label="Node Index")
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
# plt.show()

gt_graph_path = "/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_roadmap_angle_fresh_432.pkl"
gt_graph = load_graph(gt_graph_path)

def plot_all_configurations(gt_graph):
    """
    Plots all configurations (nodes) in the roadmap on a single plot.
    """
    selected_indices = [3, 4, 6, 7, 8]
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Loop through all nodes in the graph
    for node in gt_graph.nodes:
        keypoints = np.array(gt_graph.nodes[node]["configuration"])  # Extract keypoints (x, y)
        # x = keypoints[:, 0]
        # y = keypoints[:, 1]
        # ax.scatter(x, y, color='blue', s=50, alpha=0.5)

        selected_keypoints = keypoints[selected_indices, :]  # Use only selected indices
        x = selected_keypoints[:, 0]
        y = selected_keypoints[:, 1]
        ax.scatter(x, y, color='#517119', s=10, alpha=0.9)
        
        # # Connect keypoints sequentially
        # for i in range(len(x) - 1):
        #     ax.plot(x[i:i+2], y[i:i+2], color='#517119', linewidth=2, alpha=0.5)

    ax.set_title("All Configurations in the Roadmap")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_xlim(0, 640)  # Adjust based on your data range
    ax.set_ylim(0, 480)  # Adjust based on your data range
    ax.grid(False)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.show()

# Call the function to plot all configurations
plot_all_configurations(gt_graph)
    
# Function to prepare data for t-SNE
def prepare_tsne_data(gt_graph, selected_indices):
    """
    Flattens selected keypoints from all nodes for t-SNE projection.
    """
    flattened_nodes = []
    for node in gt_graph.nodes:
        keypoints = np.array(gt_graph.nodes[node]["configuration"])
        selected_keypoints = keypoints[selected_indices, :]  # Filter keypoints by selected indices
        flattened_nodes.append(selected_keypoints.flatten())  # Flatten selected keypoints into a 1D vector
    return np.array(flattened_nodes)  # Convert to NumPy array

# Selected keypoint indices
selected_indices = [3, 4, 6, 7, 8]

# Prepare data
flattened_nodes = prepare_tsne_data(gt_graph, selected_indices)

# # Perform t-SNE projection
# tsne = TSNE(n_components=2, random_state=42)
# nodes_tsne = tsne.fit_transform(flattened_nodes)

# # Plot t-SNE projection
# plt.figure(figsize=(10, 8))
# plt.scatter(nodes_tsne[:, 0], nodes_tsne[:, 1], s=5, alpha=0.7, c=range(len(nodes_tsne)), cmap='viridis')
# plt.title("Roadmap Visualization in 2D (t-SNE Projection)")
# plt.colorbar(label="Node Index")
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
# plt.grid(True)
# plt.show()

# Apply PCA
pca = PCA(n_components=2)
nodes_2d = pca.fit_transform(flattened_nodes)

# Plot the 2D projection
plt.scatter(nodes_2d[:, 0], nodes_2d[:, 1], s=50, alpha=0.7)
plt.title("Roadmap Visualization in 2D (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid()
# plt.gca().invert_yaxis()
plt.show()



# Reconstruct keypoints from the reduced dimensions
reconstructed_nodes = pca.inverse_transform(nodes_2d)  # Back to 10 dimensions

# Compare original and reconstructed keypoints for a node
original_node = flattened_nodes[0]  # Original keypoints for a node (10D)
reconstructed_node = reconstructed_nodes[0]  # Reconstructed keypoints for the same node (10D)

# Print both to compare
print("Original Keypoints (Flattened):", original_node)
print("Reconstructed Keypoints (Flattened):", reconstructed_node)

# Calculate reconstruction error
reconstruction_error = np.mean((flattened_nodes - reconstructed_nodes) ** 2)
print("Mean Reconstruction Error:", reconstruction_error)

# Example metadata: distance between first and last keypoints
metadata = [
    np.linalg.norm(flattened_nodes[i][:2] - flattened_nodes[i][-2:])  # Distance between first and last keypoints
    for i in range(flattened_nodes.shape[0])
]

# Normalize metadata for coloring
metadata_normalized = (metadata - np.min(metadata)) / (np.max(metadata) - np.min(metadata))

# Plot PCA projection with metadata overlay
plt.figure(figsize=(10, 8))
scatter = plt.scatter(nodes_2d[:, 0], nodes_2d[:, 1], c=metadata_normalized, s=50, cmap='viridis', alpha=0.8)
plt.title("PCA Projection with Metadata Overlay (Distance Between Keypoints)")
plt.colorbar(scatter, label="Normalized Metadata (e.g., Distance)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid()
plt.gca().invert_yaxis()
plt.show()
