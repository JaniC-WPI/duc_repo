#!/usr/bin/env python3

import json
import numpy as np
import cv2
import heapq
import os
import networkx as nx
import time
from sklearn.neighbors import KDTree
import shapely.geometry as geom


# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
SAFE_DISTANCE = 100  # Safe distance from the obstacle

# Load keypoints from JSON files in a given directory
def load_keypoints_from_json(directory):
    configurations = []
    for filename in os.listdir(directory):
        # if filename.endswith('.json'):
        if filename.endswith('.json') and not filename.endswith('_combined.json') and not filename.endswith('_vel.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                # Convert keypoints to integers
                keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]  # Extracting x, y coordinates
                configurations.append(np.array(keypoints))
    # print(configurations)
    return configurations

def load_and_sample_configurations(directory, num_samples):
    # Load configurations from JSON files
    configurations = load_keypoints_from_json(directory)

    # If there are more configurations than needed, sample a subset
    if len(configurations) > num_samples:
        sampled_indices = np.random.choice(len(configurations), size=num_samples, replace=False)
        sampled_configurations = [configurations[i] for i in sampled_indices]
    else:
        sampled_configurations = configurations

    return sampled_configurations


def distance_line_to_point(p1, p2, point):
    """Calculates the distance of a point to a line segment."""
    numerator = np.abs((p2[1] - p1[1]) * point[0] - (p2[0] - p1[0]) * point[1] + p2[0] * p1[1] - p2[1] * p1[0])
    denominator = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    return numerator / denominator

def square_obstacle(center, half_diagonal):
    """Creates a Shapely square polygon representing the obstacle."""
    dx = dy = half_diagonal
    x0, y0 = center[0] - dx, center[1] - dy  # Bottom-left corner
    return geom.Polygon(((x0, y0), (x0 + 2*dx, y0), (x0 + 2*dx, y0 + 2*dy), (x0, y0 + 2*dy)))

def is_collision_free(configuration, obstacle_center, safe_distance, half_diagonal):
    obstacle = square_obstacle(obstacle_center, half_diagonal + safe_distance)

    for i in range(len(configuration) - 1):
        line_segment = geom.LineString([configuration[i], configuration[i + 1]])
        if line_segment.distance(obstacle) <= 0:  # Collision! 
            return False

    return True  # Collision-free

def build_roadmap_kd_tree(configurations, start_config, goal_config, k, obstacle_center, safe_distance, half_diagonal):
    G = nx.Graph()
    # print(start_config, goal_config)
    all_configs = [start_config, goal_config] + configurations
    # print(all_configs[0])
    all_configs_np = np.array([c.flatten() for c in all_configs])

    # Add start and goal configurations to the graph
    G.add_node(tuple(map(tuple, start_config)))
    G.add_node(tuple(map(tuple, goal_config)))

    # Check if the start and goal nodes are in the graph
    print("Start node in graph:", G.has_node(tuple(map(tuple, start_config))))
    print("Goal node in graph:", G.has_node(tuple(map(tuple, goal_config))))

    # Create KDTree for efficient nearest neighbor search
    tree = KDTree(all_configs_np)

    for i, config in enumerate(all_configs):
         # Reshape the config to be 2D array for KDTree query
        config_reshaped = config.flatten().reshape(1, -1)
        # Query the k nearest neighbors
        distances, indices = tree.query(config_reshaped, k=k+1)  # k+1 because the query point itself is included

        for j in range(1, k+1):  # Skip the first index (itself)
            neighbor_config = all_configs[indices[0][j]]
            if is_collision_free(np.vstack([config, neighbor_config]), obstacle_center, safe_distance, half_diagonal):
                G.add_edge(tuple(map(tuple, config)), tuple(map(tuple, neighbor_config)), weight=distances[0][j])

    print("Final check - Start node in graph:", G.has_node(tuple(map(tuple, start_config))))
    print("Final check - Goal node in graph:", G.has_node(tuple(map(tuple, goal_config))))
    return G

def find_path_prm(graph, start_config, goal_config):
    print(graph.has_node(start_config), graph.has_node(goal_config))
    # Convert configurations to tuple for graph compatibility
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
    start_config = np.array([[272, 437], [266, 314], [175, 261], [187, 236], [230, 108], [215, 85]]) 
    goal_config = np.array([[271, 436], [267, 313], [223, 213], [248, 199], [383, 169], [404, 147]]) 
    # goal_config = np.array([[258, 367], [258, 282], [303, 217], [320, 229], [403, 283], [389, 297]])  
    # goal_config = np.array([[257, 366], [257, 283], [183, 254], [191, 234], [287, 212], [309, 216]])

    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/path_planning_panda_regression/'  # Replace with the path to your JSON files
    # configurations = load_keypoints_from_json(directory)

    # Load and sample configurations from JSON files
    num_samples = 5000 # Adjust as needed
    configurations = load_and_sample_configurations(directory, num_samples)


    # Detect the obstacle (green rectangle)
    # image_path = '/home/jc-merlab/.ros/dl_published_goal_image.jpg'  # Replace with the path to your image file
    # obstacle_info = detect_red_ball(image_path)
    # if obstacle_info is not None:
    #     obstacle_center, obstacle_radius = obstacle_info[:2], obstacle_info[2]
    # else:
    #     print("No red ball detected in the image.")
    #     obstacle_center, obstacle_radius = None, None

    obstacle_center = (400, 53)
    half_diagonal = 20
    safe_distance = half_diagonal + SAFE_DISTANCE 

    # Parameters for PRM
    num_neighbors = 500  # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    roadmap = build_roadmap_kd_tree(configurations, start_config, goal_config, num_neighbors, obstacle_center, safe_distance, half_diagonal)
    end_time = time.time()

    print("time taken to plan path", end_time - start_time)

    # Find the path
    path = find_path_prm(roadmap, start_config, goal_config)

    # path directory
    output_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/physical_path_planning/scenarios/phys_path_scene_06'

    image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/physical_path_planning/scenarios/obstacle_image_05.png'

    if path:
        print("Path found:", path)
        plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)
    else:
        print("No path found")

    # Plotting the path if found
    if path:
        print("Path found:", path)
        # plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)
    else:
        print("No path found")

    # end_time = time.time()


