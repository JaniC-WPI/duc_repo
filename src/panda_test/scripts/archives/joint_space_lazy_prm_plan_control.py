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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
SAFE_DISTANCE = 30  # Safe distance from the obstacle

def load_keypoints_from_json(directory):
    configurations = []
    configuration_ids = []
    for filename in os.listdir(directory):
        if filename.endswith('.json') and not filename.endswith('_joint_angles.json') and not filename.endswith('_vel.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]  # Extracting x, y coordinates
                configurations.append(np.array(keypoints))
                configuration_ids.append(data['id'])  # Store the configuration ID
    return configurations, configuration_ids

def load_joint_angles_from_json(directory):
    joint_angles_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith('_joint_angles.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                joint_angles_dict[data['id']] = np.array(data['joint_angles'])
    return joint_angles_dict

def skip_configurations(configurations, configuration_ids, skip_step=5, start=1, end=13000):
    skipped_configs = configurations[start:end:skip_step]
    skipped_ids = configuration_ids[start:end:skip_step]
    return skipped_configs, skipped_ids

def build_lazy_roadmap_with_kdtree(configurations, configuration_ids, joint_angles_dict, k_neighbors):
    """
    Build a LazyPRM roadmap using a KDTree for efficient nearest neighbor search based on joint angles.
    
    Args:
    - configurations: List[np.array], a list of configurations (keypoints flattened).
    - configuration_ids: List[int], a list of configuration IDs corresponding to each configuration.
    - joint_angles_dict: Dict, a dictionary mapping configuration ids to their joint angles.
    - k_neighbors: int, the number of neighbors to connect to each node.
    
    Returns:
    - G: nx.Graph, the constructed roadmap.
    - tree: KDTree, the KDTree built from joint angles.
    """
    # Flatten the configurations
    flattened_configs = np.vstack([config.flatten() for config in configurations])
    
    # Extract joint angles using the configuration IDs
    joint_angles_list = [joint_angles_dict[config_id] for config_id in configuration_ids]
    joint_angles_array = np.vstack(joint_angles_list)

    print("joint_angles", joint_angles_array)
    
    # Build KDTree using joint angles
    tree = KDTree(joint_angles_array)

    print("tree", tree)

    G = nx.Graph()
    # for i, (config, config_id) in enumerate(zip(configurations, configuration_ids)):
    #     G.add_node(i, configuration=config, joint_angles=joint_angles_dict[config_id])

    for i, (config, config_id) in enumerate(zip(configurations, configuration_ids)):
        G.add_node(i, configuration=config)

    # Query KDTree for nearest neighbors
    for i, joint_angles in enumerate(joint_angles_array):
        distances, indices = tree.query([joint_angles], k=k_neighbors + 1)  # +1 to include the node itself
        for j in indices[0]:
            if i != j:  # Avoid self-loops
                joint_angles_i = joint_angles_array[i]
                joint_angles_j = joint_angles_array[j]
                
                # Calculate the joint displacement (distance) between two configurations
                joint_displacement = np.linalg.norm(joint_angles_i - joint_angles_j)

                # Only add edge if there is no collision (collision check outside this function)
                G.add_edge(i, j, weight=joint_displacement)  # Use joint displacement as edge weight
                
                # Debugging statement to trace edge creation
                print(f"Edge added between Node {i} and Node {j} with joint displacement {joint_displacement}")

    pos_dict = {n[0]:n[1]["configuration"][5] for n in G.nodes.items()}      
    # print(pos_dict) 
    nx.draw_networkx(G,node_size=5,width=0.3, with_labels=False, pos=pos_dict)
    plt.show() 
    return G, tree

def add_config_to_roadmap_with_joint_angles(config, joint_angles, G, tree, k_neighbors, obstacle_center, half_diagonal, safe_distance):
    """
    Add a configuration with given joint angles to the roadmap using KDTree for nearest neighbor search
    and include collision checking before adding edges.
    
    Args:
    - config: np.array, the configuration (keypoints) to add.
    - joint_angles: np.array, the joint angles corresponding to the configuration.
    - G: nx.Graph, the existing roadmap graph.
    - tree: KDTree, the KDTree built from joint angles.
    - k_neighbors: int, the number of neighbors to connect to the new node.
    - obstacle_center: tuple, the (x, y) center of the obstacle.
    - half_diagonal: float, half the diagonal length of the obstacle square.
    - safe_distance: float, additional safety distance around the obstacle.
    
    Returns:
    - node_id: int, the ID of the newly added node in the roadmap.
    """
    flattened_config = config.flatten().reshape(1, -1)

    # Query KDTree for nearest neighbors based on joint angles
    distances, indices = tree.query([joint_angles], k=k_neighbors)

    node_id = len(G.nodes)

    G.add_node(node_id, configuration=config)


    # Iterate through each neighbor found by the KDTree
    for idx, dist in zip(indices[0], distances[0]):
        neighbor_config = G.nodes[idx]['configuration']

        # Retrieve the joint angles for the neighbor configuration using joint_angles_dict
        neighbor_joint_angles = joint_angles_dict[configuration_ids[idx]]

        # Check for collisions between configurations before adding an edge
        if is_collision_free(config, neighbor_config, obstacle_center, half_diagonal, safe_distance):
            # Calculate the joint displacement (distance) between the new configuration's joint angles and the neighbor's joint angles
            joint_displacement = np.linalg.norm(joint_angles - neighbor_joint_angles)

            # Add the edge with the computed joint displacement as the weight
            G.add_edge(node_id, idx, weight=joint_displacement)
            print(f"Edge added between new node {node_id} and existing node {idx} with joint displacement {joint_displacement}")
        else:
            print(f"Collision detected when attempting to connect new node {node_id} to existing node {idx}")

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
            print(f"Removed invalid edge: {u} <-> {v}")

def visualize_interactions(config1, config2, obstacle_boundary):
    fig, ax = plt.subplots()
    # Plot obstacle boundary
    obstacle_patch = PolygonPatch(obstacle_boundary, alpha=0.5, color="red", zorder=2)
    ax.add_patch(obstacle_patch)
    
    # Set plot limits and aspect
    ax.set_xlim([0, IMAGE_WIDTH])
    ax.set_ylim([IMAGE_HEIGHT, 0])
    ax.set_aspect('equal')

    # Visualize interactions within each configuration
    for config in [config1, config2]:
        for i in range(len(config) - 1):
            x_values, y_values = zip(*config[i:i+2])
            ax.plot(x_values, y_values, "blue", linewidth=2, solid_capstyle='round', zorder=1)

    # Visualize interactions between corresponding keypoints across configurations
    for i in range(len(config1)):
        x_values = [config1[i][0], config2[i][0]]
        y_values = [config1[i][1], config2[i][1]]
        ax.plot(x_values, y_values, "green", linewidth=2, linestyle="--", zorder=1)

    plt.show()
    
def visualize_interactions_path(configurations, obstacle_boundary):
    fig, ax = plt.subplots()
    # Plot obstacle boundary
    ax.add_patch(PolygonPatch(obstacle_boundary, alpha=0.5, color="red", zorder=2))
    
    # Set plot limits and aspect
    ax.set_xlim([0, IMAGE_WIDTH])
    ax.set_ylim([IMAGE_HEIGHT, 0])
    ax.set_aspect('equal')

    # Visualize paths within configurations
    for config in configurations:
        for i in range(len(config) - 1):
            segment = [config[i], config[i + 1]]
            line = geom.LineString(segment)

            # Draw the line segment
            x, y = line.xy
            ax.plot(x, y, "blue", linewidth=2, solid_capstyle='round', zorder=1)

            # Highlight if the segment intersects the obstacle
            if line.intersects(obstacle_boundary):
                ax.plot(x, y, "red", linewidth=3, solid_capstyle='round', zorder=1)

    # Visualize connections between corresponding keypoints of consecutive configurations
    for i in range(len(configurations) - 1):
        for k in range(len(configurations[i])):
            start_point = configurations[i][k]
            end_point = configurations[i + 1][k]
            line = geom.LineString([start_point, end_point])

            # Draw the connection line
            x, y = line.xy
            ax.plot(x, y, "green", linewidth=1, linestyle='--', zorder=1)

            # Highlight if the connection intersects the obstacle
            if line.intersects(obstacle_boundary):
                ax.plot(x, y, "orange", linewidth=2, linestyle='--', zorder=1)

    plt.show()

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
                # print("collision detected")
                # If any segment intersects, the configuration is not collision-free
                return False
        
    for i in range(len(configuration1)):
        segment = geom.LineString([configuration1[i], configuration2[i]])
        if segment.intersects(obstacle_boundary):
            print("edge collision detected")
            return False
        
     # If no segments intersect, the configuration is collision-free
    return True

def find_path(G, start_node, goal_node):
    path_indices = nx.astar_path(G, source=start_node, target=goal_node)
    path_configurations = [G.nodes[i]['configuration'] for i in path_indices]

    return path_configurations

def plot_path_on_image_dir(image_path, path, start_config, goal_config, output_directory):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Base image
    base_image = cv2.imread(image_path)

    # Function to generate a color
    def path_color():
        return (255, 0, 0)  # Blue color for path

    # Draw start and goal keypoints
    for point in start_config:
        cv2.circle(base_image, tuple(point.astype(int)), radius=5, color=(0, 0, 255), thickness=-1)  # Red for start
    for point in goal_config:
        cv2.circle(base_image, tuple(point.astype(int)), radius=5, color=(0, 255, 0), thickness=-1)  # Green for goal

    # Draw and save each path configuration
    for idx, config_tuples in enumerate(path):
        image = base_image.copy()  # Copy the base image
        config = np.array(config_tuples)  # Convert config from tuple of tuples to numpy array
        for i in range(len(config) - 1):
            cv2.line(image, tuple(config[i].astype(int)), tuple(config[i+1].astype(int)), path_color(), 2)
        for point in config:
            cv2.circle(image, tuple(point.astype(int)), radius=6, color=path_color(), thickness=-1)        
        
        # Save the image
        cv2.imwrite(os.path.join(output_directory, f'path_{idx}.jpg'), image)

# Main execution
if __name__ == "__main__":
    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged/'  # Replace with the path to your JSON files
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b128_e500_v34.pth'
    num_samples = 500

    skip_step = 10
    start_index = 1
    end_index = 25000

    configurations, configuration_ids = load_keypoints_from_json(directory)
    joint_angles_dict = load_joint_angles_from_json(directory)

    skipped_configs, skipped_ids = skip_configurations(configurations, configuration_ids, skip_step, start_index, end_index)

    # Parameters for PRM
    num_neighbors = 25 # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    roadmap, tree = build_lazy_roadmap_with_kdtree(skipped_configs, skipped_ids, joint_angles_dict, num_neighbors)
    end_time = time.time()

    print("time taken to find the graph", end_time - start_time)      

    # Define start and goal configurations as numpy arrays
    start_config = np.array([[250, 442], [252, 311], [215, 273], [172, 234], [192, 212], [220, 147], [249, 82], [248, 52], [286, 48]])
    goal_config = np.array([[250, 442], [252, 311], [275, 255], [294, 200], [322, 209], [394, 194], [468, 181], [494, 158], [522, 187]])

    start_joint_angles = np.array([0.9331, -1.33819, 2.2474])    
    goal_joint_angles = np.array([0.267307, -1.38323, 2.58668])

    SAFE_ZONE = 40  # Safe distance from the obstacle
    obstacle_center = (350, 120)
    half_diagonal = 20

    # Add start configuration to roadmap
    start_node = add_config_to_roadmap_with_joint_angles(start_config, start_joint_angles, roadmap, tree, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE)
    goal_node = add_config_to_roadmap_with_joint_angles(goal_config, goal_joint_angles, roadmap, tree, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE)
    
    validate_and_remove_invalid_edges(roadmap, obstacle_center, half_diagonal, SAFE_ZONE)
        
    # Find and print the path from start to goal
    path = find_path(roadmap, start_node, goal_node)

    if path:
         point_set = []
         goal_sets = []
         # Iterate through the path, excluding the first and last configuration
         for configuration in path[0:-1]:
            # Extract the last three keypoints of each configuration
            selected_points = configuration[[3, 4, 6, 7, 8]]
            selected_points_float = [[float(point[0]), float(point[1])] for point in selected_points]
            # Append these points to the point_set list
            point_set.append(selected_points_float)

         # Iterate through the path, excluding start and goal            
         for configuration in path[1:]: 
            selected_points = configuration[[3, 4, 6, 7, 8]]
            selected_points_float = [[float(point[0]), float(point[1])] for point in selected_points]
            goal_features = []
            for point in selected_points_float:
                goal_features.extend(point)  # Add x, y as a pair
            goal_sets.append(goal_features)

         print("Point Set:", point_set)
         print("goal sets: ", goal_sets)
    
         with open("/home/jc-merlab/duc_repo/src/panda_test/config/dl_multi_features.yaml", "w") as yaml_file:
             s = "dl_controller:\n"
             s += "  num_goal_sets: " + str(len(goal_sets)) + "\n"
             for i, goal in enumerate(goal_sets, start=1):
                 # Convert the list of floats into a comma-separated string
                 goal_str = ', '.join(map(str, goal))
                 s += f"  goal_features{i}: [{goal_str}]\n"
    
             # Write the string to the file
             yaml_file.write(s)

         with open("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space/42/dl_multi_features.yaml", "w") as yaml_file:
             s = "dl_controller:\n"
             s += "  num_goal_sets: " + str(len(goal_sets)) + "\n"
             for i, goal in enumerate(goal_sets, start=1):
                 # Convert the list of floats into a comma-separated string
                 goal_str = ', '.join(map(str, goal))
                 s += f"  goal_features{i}: [{goal_str}]\n"
    
             # Write the string to the file
             yaml_file.write(s)
    
         print("Data successfully written to config/dl_multi_features.yaml")

         # Save configurations to a .txt file
         with open("/home/jc-merlab/duc_repo/src/panda_test/config/path_configurations.txt", "w") as file:
             file.write("Start Configuration:\n")
             file.write(str(start_config.tolist()) + "\n\n")
             file.write("Goal Configuration:\n")
             file.write(str(goal_config.tolist()) + "\n\n")
             file.write("Obstacle Parameters:\n")
             file.write("Safe Zone:\n")
             file.write(str(SAFE_ZONE) + "\n\n")
             file.write("Obstacle Center:\n")
             file.write(str(obstacle_center) + "\n\n")
             file.write("Half Diagonal:\n")
             file.write(str(half_diagonal) + "\n\n")
             file.write("Path:\n")
             for config in path:
                 file.write(str(config.tolist()) + "\n")
             file.write("\nPoint Set:\n")
             for points in point_set:
                 file.write(str(points) + "\n")

         with open("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space/42/path_configurations.txt", "w") as file:
             file.write("Start Configuration:\n")
             file.write(str(start_config.tolist()) + "\n\n")
             file.write("Goal Configuration:\n")
             file.write(str(goal_config.tolist()) + "\n\n")
             file.write("Obstacle Parameters:\n")
             file.write("Safe Zone:\n")
             file.write(str(SAFE_ZONE) + "\n\n")
             file.write("Obstacle Center:\n")
             file.write(str(obstacle_center) + "\n\n")
             file.write("Half Diagonal:\n")
             file.write(str(half_diagonal) + "\n\n")
             file.write("Path:\n")
             for config in path:
                 file.write(str(config.tolist()) + "\n")
             file.write("\nPoint Set:\n")
             for points in point_set:
                 file.write(str(points) + "\n")

         print("Configurations successfully saved to configurations.txt")

    
    

