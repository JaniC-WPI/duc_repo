#!/usr/bin/env python3

import json
import numpy as np
import cv2
import heapq
import os
import networkx as nx
import time
from sklearn.neighbors import KDTree


# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
SAFE_DISTANCE = 50  # Safe distance from the obstacle

# Load keypoints from JSON files in a given directory
def load_keypoints_from_json(directory):
    configurations = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                # Convert keypoints to integers
                keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]  # Extracting x, y coordinates
                configurations.append(np.array(keypoints))
    # print(configurations)
    return configurations

# Detect a red ball in an image
def detect_red_ball(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Red color might be in two ranges
    lower_red_1 = np.array([0, 50, 50])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 50, 50])
    upper_red_2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = mask1 + mask2  # Combine masks
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        return (int(x), int(y), int(radius))
    return None

def line_points(p1, p2, interval=10):
    """Return all the points on the line segment from p1 to p2 using Bresenham's Line Algorithm."""
    x1, y1 = p1
    x2, y2 = p2
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x2:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    # print(points)
    return points[::interval]

def is_line_too_close_to_obstacle(p1, p2, obstacle_center, safe_distance):
    """Check if a line segment between p1 and p2 is too close to the obstacle."""
    for point in line_points(p1, p2):
        print(point)
        print(np.linalg.norm(np.array(point) - np.array(obstacle_center)))
        print(np.array(point) - np.array(obstacle_center))
        print(obstacle_center)
        if np.linalg.norm(np.array(point) - np.array(obstacle_center)) < safe_distance:
            return True
    return False

def is_collision_free(configuration, obstacle_center, safe_distance):
    for i in range(len(configuration) - 1):
        if is_line_too_close_to_obstacle(configuration[i], configuration[i + 1], obstacle_center, safe_distance):
            return False
    return True

def is_collision_free_line_check(configuration, obstacle_center, safe_distance):
    # Define a function to check for collision along a line segment
    def is_line_colliding(p1, p2, obstacle_center, safe_distance):
        # Check for collision at regular intervals along the line
        num_checks = int(np.linalg.norm(np.array(p2) - np.array(p1)) / safe_distance)
        for i in range(num_checks + 1):
            point = np.array(p1) + (np.array(p2) - np.array(p1)) * i / num_checks
            if np.linalg.norm(point - np.array(obstacle_center)) < safe_distance:
                return True
        return False

    for i in range(len(configuration) - 1):
        if is_line_colliding(configuration[i], configuration[i + 1], obstacle_center, safe_distance):
            return False
    return True


def sample_configurations(num_samples, image_width, image_height, obstacle_center, safe_distance):
    configurations = []
    while len(configurations) < num_samples:
        config = np.random.uniform(low=[0, 0], high=[image_width, image_height], size=(6, 2)).astype(int)
        if is_collision_free(config, obstacle_center, safe_distance):
            configurations.append(config)
    return configurations

def build_roadmap(configurations, k, obstacle_center, safe_distance):
    G = nx.Graph()
    for config in configurations:
        G.add_node(tuple(map(tuple, config)))

    for i, config1 in enumerate(configurations):
        # Connect to k-nearest neighbors
        distances = [np.linalg.norm(config1 - config2) for config2 in configurations]
        nearest_indices = np.argsort(distances)[1:k+1]
        for j in nearest_indices:
            config2 = configurations[j]
            if is_collision_free(np.vstack([config1, config2]), obstacle_center, safe_distance):
                G.add_edge(tuple(map(tuple, config1)), tuple(map(tuple, config2)), weight=distances[j])
    return G

def build_roadmap_kd_tree(configurations, start_config, goal_config, k, obstacle_center, safe_distance):
    G = nx.Graph()
    all_configs = [start_config, goal_config] + configurations
    all_configs_np = np.array([c.flatten() for c in all_configs])

    # Create KDTree for efficient nearest neighbor search
    tree = KDTree(all_configs_np)

    for i, config in enumerate(all_configs):
         # Reshape the config to be 2D array for KDTree query
        config_reshaped = config.flatten().reshape(1, -1)
        # Query the k nearest neighbors
        distances, indices = tree.query(config_reshaped, k=k+1)  # k+1 because the query point itself is included

        for j in range(1, k+1):  # Skip the first index (itself)
            neighbor_config = all_configs[indices[0][j]]
            if is_collision_free_line_check(np.vstack([config, neighbor_config]), obstacle_center, safe_distance):
                G.add_edge(tuple(map(tuple, config)), tuple(map(tuple, neighbor_config)), weight=distances[0][j])

    return G


def find_path_prm(graph, start_config, goal_config):
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
    # Define the start and goal configurations (generalized for n keypoints)
    start_config = np.array([[257, 366], [257, 283], [179, 297], [175, 276], [175, 177], [197, 181]])  
    goal_config = np.array([[257, 366], [257, 283], [303, 217], [320, 229], [403, 283], [389, 297]])  
    # goal_config = np.array([[257, 366], [257, 283], [183, 254], [191, 234], [287, 212], [309, 216]])

    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/kprcnn_sim_latest/'  # Replace with the path to your JSON files
    configurations = load_keypoints_from_json(directory)

    # Detect the obstacle (red ball)
    image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/red_ball_image_1_goal.jpg'  # Replace with the path to your image file
    obstacle_info = detect_red_ball(image_path)
    if obstacle_info is not None:
        obstacle_center, obstacle_radius = obstacle_info[:2], obstacle_info[2]
    else:
        print("No red ball detected in the image.")
        obstacle_center, obstacle_radius = None, None

    # Parameters for PRM
    num_neighbors = 50  # Number of neighbors for each node in the roadmap

    start_time = time.time()   
    # Build the roadmap
    roadmap = build_roadmap_kd_tree(configurations, start_config, goal_config, num_neighbors, obstacle_center, SAFE_DISTANCE)
    end_time = time.time()
    
    print("time taken to build roadmap", end_time - start_time)

    # Find the path
    path = find_path_prm(roadmap, start_config, goal_config)

    # path directory
    output_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/paths/path_3_prm'

    # Plotting the path if found
    if path:
        print("Path found:", path)
        plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)
    else:
        print("No path found")


# [
#                 257.95220042652915,
#                 366.9198630617724,
#                 1
#             ]
#         ],
#         [
#             [
#                 257.95973939799904,
#                 283.013113744617,
#                 1
#             ]
#         ],
#         [
#             [
#                 212.68375962913188,
#                 217.47046343476333,
#                 1
#             ]
#         ],
#         [
#             [
#                 229.81850714994908,
#                 205.66649652451525,
#                 1
#             ]
#         ],
#         [
#             [
#                 327.56421488703063,
#                 190.11205592376203,
#                 1
#             ]
#         ],
#         [
#             [
#                 349.5370638862348,
#                 192.6387930737868,
#                 1
#             ]
#         ]



