import pickle
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from itertools import islice
import pandas as pd

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
    ax.scatter(joint_1, joint_2, joint_3, c='b', marker='_')

    ax.set_xlabel('Joint 1', fontsize=10, fontweight='bold')
    ax.set_ylabel('Joint 2', fontsize=10, fontweight='bold')
    ax.set_zlabel('Joint 3', fontsize=10, fontweight='bold')

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
    Plot histograms of joint angle distances for multiple datasets with same bin width.

    Args:
    - distance_list (list of lists): List of joint angle distances for each dataset.
    - labels (list of str): List of labels for the datasets.
    - output_prefix (str): Prefix for saving the output plot files.
    - fontsize (int): Font size for plot labels and titles.

    Returns:
    - None
    """
    plt.figure(figsize=(15, 5))  # Create a figure for the histograms

    
    # Determine the common range across all datasets (min and max distances)
    min_dist = min(min(distances) for distances in distance_list)
    max_dist = max(max(distances) for distances in distance_list)
    
    
    print(min_dist, max_dist)
    
    # Store bin information for later analysis
    max_bin_info = {}

    # Plot histograms for the datasets
    for i, distances in enumerate(distance_list):
        plt.subplot(1, 3, i + 1)
        
        # Determine the common range across all datasets (min and max distances)
        min_distance = min(distances) 
        max_distance = max(distances)

        # Define a consistent bin width (e.g., 0.1 or any suitable width based on the data)
        bin_width = 0.1  # You can adjust this value based on your needs
        bin_edges = np.arange(min_distance, max_distance, bin_width)   
        
        print("number of bins", len(bin_edges))
        
        counts, bin_edges = np.histogram(distances, bins=bin_edges, range=(min_dist, 2.5))
        
        # Find the bin with the maximum number of data points
        max_bin_index = np.argmax(counts)
        max_bin_count = counts[max_bin_index]
        max_bin_range = (bin_edges[max_bin_index], bin_edges[max_bin_index + 1])

        # Store the max bin information
        max_bin_info[labels[i]] = {
            'max_bin_count': max_bin_count,
            'max_bin_range': max_bin_range
        }
        
        print(max_bin_info)

        # Plot the histogram for the current dataset using the same bin edges
        plt.hist(distances, bins=bin_edges, alpha=0.7, color=colors[i])
        
        plt.xlim(min_dist, 2.5)
        plt.ylim(0, 21000)

        plt.xticks(fontsize=fontsize, weight='bold')
        plt.yticks(fontsize=fontsize, weight='bold')

        # plt.title(f'{labels[i]}', fontsize=fontsize + 2)
        # plt.xlabel('Joint Distance', fontsize=fontsize)
        # plt.ylabel('Number of Connected Edges', fontsize=fontsize)
        # plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_joint_angle_distances_same_bin_width.png')
    plt.show()
    
def plot_joint_distance_boxplots_with_stats(distance_list, labels, colors, output_path, fontsize=12, showfliers=False):
    """
    Plot box plots of joint angle distances for multiple datasets and display mean, std dev, and median.
    Ensure that annotations are visible even when outliers are not displayed.
    """
    plt.figure(figsize=(12, 6))  # Set the figure size

    # Create the box plot with or without outliers
    box = plt.boxplot(distance_list, patch_artist=True, notch=True, vert=True, showfliers=showfliers)

    # Set colors for each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Calculate the overall range of the data (including outliers) for setting the y-axis
    all_data = np.concatenate(distance_list)
    y_upper_limit = max(all_data)*0.9  # Add some margin for annotations
    # y_lower_limit = min(all_data)  # Add margin for the bottom

    # Calculate and display mean, std dev, and median for each dataset
    for i, distances in enumerate(distance_list):
        mean_val = np.mean(distances)
        std_dev_val = np.std(distances)
        median_val = np.median(distances)

        # Adjust position for each annotation based on dataset
        plt.text(i + 1.0, y_upper_limit*.55, f'Mean: {mean_val:.4f}\nStd: {std_dev_val:.4f}', 
                 ha='center', va='top', fontsize=fontsize - 2, color='black', fontweight='bold')

        # Annotate the median value on the box plot
        plt.text(i + 1, median_val, f'Median: {median_val:.4f}', 
                 ha='center', va='center', fontsize=fontsize - 1, color='white', fontweight='bold')

    # Set the y-axis limits manually to ensure visibility of the annotations
    # plt.ylim(y_lower_limit, y_upper_limit)

    # Set labels and title
    plt.xticks(ticks=[1, 2, 3], labels=labels, fontsize=fontsize, fontweight='bold')
    plt.yticks(fontsize=fontsize, weight='bold')

    # plt.xlabel('Roadmaps', fontsize=fontsize)
    # plt.ylabel('Joint Angle Distance', fontsize=fontsize)
    # plt.title('Box Plot of Joint Angle Distances with Mean, Std Dev, and Median', fontsize=fontsize)

    # Save and show the plot
    plt.savefig(output_path)
    plt.show()    


def plot_bar_comparison(csv_file, output_image_path):
    """
    Creates a bar plot showing the count of trials where each method has the largest total distance.
    
    Args:
    - csv_file (str): Path to the CSV file containing the total joint distances for each trial.
    - output_image_path (str): Path to save the output bar plot image.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Count how many times each method has the largest total distance
    largest_counts = {'Ground Truth': 0, 'Custom': 0, 'Euclidean': 0}
    for _, row in df.iterrows():
        max_method = row[['Ground Truth', 'Custom', 'Euclidean']].idxmax()
        largest_counts[max_method] += 1

    # Create a bar plot
    methods = list(largest_counts.keys())
    counts = list(largest_counts.values())

    plt.figure(figsize=(8, 6))
    plt.bar(methods, counts, color=['green', 'blue', 'red'])

    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, weight='bold')

    # Set plot labels and title
    # plt.ylabel('Count of Trials with Largest Distance')
    # plt.title('Comparison of Methods by Count of Trials with Largest Distance')

    # Save the plot as an image
    plt.savefig(output_image_path)
    plt.show()

def main():
    # Paths to the saved roadmap files
    custom_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_roadmap_angle_fresh_432_edges.pkl'
    euclidean_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_roadmap_angle_fresh_432_edges.pkl'
    joint_space_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space_roadmap_angle_fresh_432_edges.pkl'

    # Load the roadmaps
    custom_graph = load_graph(custom_graph_path)
    euclidean_graph = load_graph(euclidean_graph_path)
    joint_space_graph = load_graph(joint_space_graph_path)

    plot_joint_angles_3d(custom_graph, 'Custom Distance', 'custom_roadmap_joint_angles.png')
    plot_joint_angles_3d(euclidean_graph, 'Euclidean Distance', 'euclidean_roadmap_joint_angles.png')
    plot_joint_angles_3d(joint_space_graph, 'Ground Truth Distance', 'gt_roadmap_joint_angles.png')

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
    # Calculate joint angle distances for edges in each roadmap
    custom_joint_distances = calculate_joint_angle_distances(custom_graph)
    euclidean_joint_distances = calculate_joint_angle_distances(euclidean_graph)
    joint_space_joint_distances = calculate_joint_angle_distances(joint_space_graph)
    

    # Ensure that distance lists are not empty
    if len(custom_joint_distances) == 0 or len(euclidean_joint_distances) == 0 or len(joint_space_joint_distances) == 0:
        print("Error: One or more joint angle distance datasets are empty. Please check the input data.")
        return

    # Plot histograms for joint angle distances
    distance_list = [joint_space_joint_distances, custom_joint_distances, euclidean_joint_distances]
    labels = ['Ground Truth Roadmap', 'Custom Roadmap', 'Euclidean Roadmap']
    colors = ['green', 'blue', 'red']
    output_prefix = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_angle_distances'
    plot_joint_distance_histograms(distance_list, labels, colors, output_prefix)
    plot_joint_distance_boxplots_with_stats(distance_list, labels, colors, output_prefix)

    # Load the CSV file
    csv_file_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/gt_fresh_trial/total_joint_angle_distances_fresh_mod_astar.csv'  # Replace with the path to your CSV file
    df = pd.read_csv(csv_file_path)

    # Create a new column that identifies which roadmap has the largest joint distance for each trial
    df['Largest Distance Roadmap'] = df[['Ground Truth', 'Custom', 'Euclidean']].idxmax(axis=1)

    # Count the occurrences of each roadmap having the largest joint distance
    roadmap_counts = df['Largest Distance Roadmap'].value_counts()

    print(roadmap_counts)

    # Define font properties
    font_properties = {'fontsize': 16, 'fontweight': 'bold', 'color':'white'}

    print(roadmap_counts.index)

    # Plot a pie chart with larger font size and bold labels
    fig, ax = plt.subplots()
    ax.pie(roadmap_counts, 
           labels=roadmap_counts.index, 
           autopct='%1.1f%%', 
           startangle=90, 
           colors=['red', 'blue', 'green'], 
           textprops=font_properties)

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    custom_legend_labels = ['Ground Truth', 'Custom', 'Euclidean']
    custom_colors = ['green', 'blue', 'red']
    patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in custom_colors]
    plt.legend(patches, custom_legend_labels, loc="best", fontsize=14, frameon=False)

    output_image_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/bar_plot_comparison_plot.png'
    # Title for the pie chart
    # plt.title('Proportion of Trials with Largest Joint Distances for Each Roadmap', fontsize=14, fontweight='bold')

    # Show the pie chart
    plt.show()

    plot_bar_comparison(csv_file_path, output_image_path)

if __name__ == "__main__":
    main()
