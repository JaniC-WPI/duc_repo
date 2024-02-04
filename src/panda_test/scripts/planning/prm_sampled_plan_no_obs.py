#!/usr/bin/env python3

#!/usr/bin/env python3

import numpy as np
import os
import json
import networkx as nx
from sklearn.neighbors import KDTree
import cv2
import time

# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480

def load_keypoints_from_json(directory):
    configurations = []
    for filename in os.listdir(directory):
        if filename.endswith('.json') and not filename.endswith('_combined.json') and not filename.endswith('_vel.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]
                configurations.append(np.array(keypoints))
    return configurations

def load_and_sample_configurations(directory, num_samples):
    configurations = load_keypoints_from_json(directory)
    if len(configurations) > num_samples:
        sampled_indices = np.random.choice(len(configurations), size=num_samples, replace=False)
        sampled_configurations = [configurations[i] for i in sampled_indices]
    else:
        sampled_configurations = configurations
    return sampled_configurations

def build_roadmap_kd_tree(configurations, start_config, goal_config, k):
    G = nx.Graph()
    all_configs = [start_config, goal_config] + configurations
    all_configs_np = np.array([c.flatten() for c in all_configs])

    G.add_node(tuple(map(tuple, start_config)))
    G.add_node(tuple(map(tuple, goal_config)))

    tree = KDTree(all_configs_np)

    for i, config in enumerate(all_configs):
        config_reshaped = config.flatten().reshape(1, -1)
        distances, indices = tree.query(config_reshaped, k=k+1)
        for j in range(1, k+1):
            neighbor_config = all_configs[indices[0][j]]
            G.add_edge(tuple(map(tuple, config)), tuple(map(tuple, neighbor_config)), weight=distances[0][j])
    return G

def find_path_prm(graph, start_config, goal_config):
    start = tuple(map(tuple, start_config))
    goal = tuple(map(tuple, goal_config))
    try:
        path = nx.astar_path(graph, start, goal)
        return path
    except nx.NetworkXNoPath:
        return None

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
            cv2.circle(image, tuple(point.astype(int)), radius=3, color=path_color(), thickness=-1)        
        
        # Save the image
        cv2.imwrite(os.path.join(output_directory, f'path_{idx}.jpg'), image)

# Main execution
if __name__ == "__main__":
    # start_time = time.time()
    # Define the start and goal configurations (generalized for n keypoints)
    # start_config = np.array([[258, 367], [258, 282], [179, 297], [175, 276], [175, 177], [197, 181]])
    start_config = np.array([[258, 367], [258, 282], [178, 282], [177, 262], [179, 164], [200, 164]])
    # goal_config = np.array ([[258, 367], [258, 282], [191, 240], [202, 223], [291, 178], [298, 196]])
    goal_config = np.array([[258, 366], [258, 281], [298, 214], [315, 224], [398, 276], [388, 292]])
    # goal_config = np.array([[258, 367], [258, 282], [303, 217], [320, 229], [403, 283], [389, 297]])  
    # goal_config = np.array([[257, 366], [257, 283], [183, 254], [191, 234], [287, 212], [309, 216]])

    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/path_planning_combined/'  # Replace with the path to your JSON files
    # configurations = load_keypoints_from_json(directory)

    # Load and sample configurations from JSON files
    num_samples = 500 # Adjust as needed
    configurations = load_and_sample_configurations(directory, num_samples)
    image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/red_ball_image_3.jpg'

    obstacle_center = (319, 138)
    obstacle_radius = 25

    # Parameters for PRM
    num_neighbors = 50  # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    roadmap = build_roadmap_kd_tree(configurations, start_config, goal_config, num_neighbors)
    end_time = time.time()

    print("time taken to plan path", end_time - start_time)

    # Find the path
    path = find_path_prm(roadmap, start_config, goal_config)

    # path directory
    output_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/paths/path_prm/path_12_prm'

    # Plotting the path if found
    if path:
        print("Path found:", path)
        # plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)
    else:
        print("No path found")

    # end_time = time.time()


