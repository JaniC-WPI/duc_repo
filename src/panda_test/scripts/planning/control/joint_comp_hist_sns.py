import pickle
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from itertools import islice
import pandas as pd
import seaborn as sns


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

def plot_joint_distance_histograms(distance_list, labels, colors, output_prefix, fontsize=12):
    """
    Plot histograms of joint angle distances for multiple datasets with same bin width using Seaborn.

    Args:
    - distance_list (list of lists): List of joint angle distances for each dataset.
    - labels (list of str): List of labels for the datasets.
    - colors (list of str): List of colors for each dataset.
    - output_prefix (str): Prefix for saving the output plot files.
    - fontsize (int): Font size for plot labels and titles.

    Returns:
    - None
    """
    plt.figure(figsize=(20, 10))  # Create a figure for the histograms

    # Determine the common range across all datasets (min and max distances)
    min_dist = min(min(distances) for distances in distance_list)
    max_dist = max(max(distances) for distances in distance_list)
    
    bin_width = 0.1  # Set bin width
    bin_edges = np.arange(min_dist, max_dist + bin_width, bin_width)  # Define bins with consistent width

    # Store bin information for later analysis
    max_bin_info = {}

    # Plot histograms for the datasets
    for i, distances in enumerate(distance_list):
        plt.subplot(1, 3, i + 1)

        # Use Pandas DataFrame to facilitate plotting with Seaborn
        df = pd.DataFrame(distances, columns=['Distance'])

        # Plot using Seaborn with specified bins
        sns.histplot(
            df['Distance'],
            bins=bin_edges,
            color=colors[i],
            kde=False,
            edgecolor="black",
            alpha=0.9
        )

        # Calculate histogram counts and bin ranges for max bin info
        counts, _ = np.histogram(distances, bins=bin_edges, range=(min_dist, 2.5))
        max_bin_index = np.argmax(counts)
        max_bin_count = counts[max_bin_index]
        max_bin_range = (bin_edges[max_bin_index], bin_edges[max_bin_index + 1])

        max_bin_info[labels[i]] = {
            'max_bin_count': max_bin_count,
            'max_bin_range': max_bin_range
        }

        print(f"{labels[i]} max bin info:", max_bin_info[labels[i]])

        # Set plot limits and labels
        plt.xlim(min_dist, 2.3)
        plt.ylim(0, 21000)
        plt.xticks(fontsize=32, weight='bold')
        # plt.yticks(fontsize=24, weight='bold')
        # Only keep y-tick labels on the first subplot
        if i == 0:
            plt.yticks(fontsize=32, weight='bold')
        else:
            plt.yticks([])  # Remove y-tick labels for other subplots
        # plt.title(f'{labels[i]}', fontsize=fontsize + 2)
        # plt.xlabel('Joint Distance', fontsize=fontsize)
        # plt.ylabel('Number of Connected Edges', fontsize=fontsize)
        plt.xlabel('')
        plt.ylabel('')

    plt.tight_layout()
    # plt.savefig(f'{output_prefix}_joint_angle_distances_same_bin_width.png')
    plt.savefig('/media/jc-merlab/Crucial X9/paper_data/all_rm_joint_hist_slide.pdf')
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


    # common_edge_custom = find_common_edges(custom_graph, joint_space_graph)
    # common_edge_euclidean = find_common_edges(euclidean_graph, joint_space_graph)

    # print(common_edge_custom, common_edge_euclidean)

    # Print graph statistics
    # print_graph_statistics(custom_graph, "Custom PRM")
    # print_graph_statistics(euclidean_graph, "Euclidean PRM")
    # print_graph_statistics(joint_space_graph, "Joint Space PRM")

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
    colors = ['#40B0A6', '#5D3A9B', '#D41159']
    output_prefix = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_angle_distances'
    plot_joint_distance_histograms(distance_list, labels, colors, output_prefix)

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
           colors=['#D41159', '#5D3A9B' , '#40B0A6'], 
           textprops=font_properties)

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    custom_legend_labels = ['Ground Truth', 'Custom', 'Euclidean']
    custom_colors = ['#40B0A6', '#5D3A9B', '#D41159']
    patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in custom_colors]
    legend_font_properties = {'weight': 'bold', 'size': 14}
    plt.legend(patches, custom_legend_labels, loc="best", prop=legend_font_properties, frameon=True)

    output_image_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/bar_plot_comparison_plot.png'
    # Title for the pie chart
    # plt.title('Proportion of Trials with Largest Joint Distances for Each Roadmap', fontsize=14, fontweight='bold')

    # Show the pie chart
    plt.show()


if __name__ == "__main__":
    main()
