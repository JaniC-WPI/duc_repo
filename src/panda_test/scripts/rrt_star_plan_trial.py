#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2

# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
MAX_ITER = 700
STEP_SIZE = 10  # Pixels
SAFE_DISTANCE = 10  # Safe distance from the obstacle

# Load keypoints from JSON files in a given directory
def load_keypoints_from_json(directory):
    configurations = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                keypoints = [point[0][:2] for point in data['keypoints']]  # Extracting x, y coordinates
                configurations.append(np.array(keypoints))
    # print(configurations)
    return configurations

# Detect a green ball in an image
def detect_green_ball(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, green_lower, green_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        return (int(x), int(y), int(radius))
    return None

# Check if a configuration is too close to the obstacle
def is_too_close_to_obstacle(configuration, obstacle_center, safe_distance):
    for point in configuration:
        if np.linalg.norm(point - obstacle_center[:2]) < safe_distance:
            return True
    return False

# RRT Node class
class Node:
    def __init__(self, config, parent=None):
        self.config = config
        self.parent = parent

# Euclidean distance between configurations
def config_distance(config1, config2):
    return np.linalg.norm(config1 - config2)

# Nearest node in the tree to the given configuration
def nearest_node(tree, config):
    return min(tree, key=lambda node: config_distance(node.config, config))

# Steer towards a sampled configuration
def steer(from_config, to_config, step_size):
    direction = to_config - from_config
    norm = np.linalg.norm(direction, axis=1)
    new_config = np.copy(from_config).astype(float)
    for i in range(len(from_config)):
        if norm[i] > step_size:
            new_config[i] += step_size * direction[i] / norm[i]
    return new_config.astype(int)

# Modified RRT algorithm
def modified_rrt(start, goal, configurations, obstacle_center, safe_distance):
    tree = [Node(start)]
    for _ in range(MAX_ITER):
        random_config = random.choice(configurations)  # Randomly choose a configuration from the dataset
        nearest = nearest_node(tree, random_config)
        new_config = steer(nearest.config, random_config, STEP_SIZE)
        if not is_too_close_to_obstacle(new_config, obstacle_center, safe_distance):
            new_node = Node(new_config, nearest)
            tree.append(new_node)
            if config_distance(new_config, goal) < STEP_SIZE:
                return new_node
    return None

# Calculate cost of the node (distance from start)
def calculate_cost(node):
    cost = 0
    current = node
    while current.parent is not None:
        cost += config_distance(current.config, current.parent.config)
        current = current.parent
    return cost

# Find nearby nodes within a specified radius
def find_nearby_nodes(tree, node, radius):
    return [other_node for other_node in tree if config_distance(node.config, other_node.config) < radius]

# Rewire the tree if a better path is found
def rewire(tree, new_node, nearby_nodes, obstacle_center, safe_distance):
    for nearby_node in nearby_nodes:
        new_cost = calculate_cost(new_node) + config_distance(new_node.config, nearby_node.config)
        if new_cost < calculate_cost(nearby_node) and not is_too_close_to_obstacle(nearby_node.config, obstacle_center, safe_distance):
            nearby_node.parent = new_node

# Modified RRT* Algorithm
def rrt_star(start, goal, configurations, obstacle_center, safe_distance, search_radius):
    tree = [Node(start)]
    for _ in range(MAX_ITER):
        random_config = random.choice(configurations)
        nearest = nearest_node(tree, random_config)
        new_config = steer(nearest.config, random_config, STEP_SIZE)
        if not is_too_close_to_obstacle(new_config, obstacle_center, safe_distance):
            new_node = Node(new_config, nearest)
            tree.append(new_node)
            nearby_nodes = find_nearby_nodes(tree, new_node, search_radius)
            rewire(tree, new_node, nearby_nodes, obstacle_center, safe_distance)
            if config_distance(new_config, goal) < STEP_SIZE:
                return new_node
    return None

# Extract path from the goal node to the start
def extract_path(goal_node):
    path = []
    current = goal_node
    while current:
        path.append(current)
        current = current.parent
    return path[::-1]

# Visualization function
def plot_path(path, start_config, goal_config, obstacle_center):
    plt.figure(figsize=(8, 6))
    if path:
        # Extract the configuration data from each Node in the path
        path_configs = [node.config for node in path]
        for i in range(len(start_config)):
            x, y = zip(*[config[i] for config in path_configs])
            plt.plot(x, y, '-o', label=f'Keypoint {i+1} Path')
    plt.scatter(*start_config.T, color='green', label='Start', s=50)
    plt.scatter(*goal_config.T, color='red', label='Goal', s=50)
    plt.scatter(*obstacle_center, color='blue', label='Obstacle', s=100)
    plt.xlim(0, IMAGE_WIDTH)
    plt.ylim(0, IMAGE_HEIGHT)
    plt.gca().invert_yaxis()
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')
    plt.title('RRT Path in Image Space')
    plt.legend()
    plt.show()

# def plot_path_on_image(image_path, path, start_config, goal_config):
#     # Load the original image
#     image = cv2.imread(image_path)

#     # Colors for the keypoints (in BGR format)
#     color_start = (0, 0, 255)  # Red
#     color_goal = (0, 255, 0)  # Green
#     color_path = (255, 0, 0)  # Blue

#     # Draw start and goal keypoints
#     for point in start_config:
#         cv2.circle(image, tuple(point.astype(int)), radius=5, color=color_start, thickness=-1)
#     for point in goal_config:
#         cv2.circle(image, tuple(point.astype(int)), radius=5, color=color_goal, thickness=-1)

#     # Draw intermediate keypoints along the path
#     if path:
#         path_configs = [node.config for node in path]
#         for config in path_configs:
#             for point in config:
#                 cv2.circle(image, tuple(point.astype(int)), radius=3, color=color_path, thickness=-1)

#     # Display the image
#     # cv2.imshow('Path Visualization', image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     # Optionally, save the image
#     save_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/path_with_keypoints.jpg'  # Modify as needed
#     cv2.imwrite(save_path, image)

# def plot_path_on_image(image_path, path, start_config, goal_config):
#     # Load the original image
#     image = cv2.imread(image_path)

#     # Colors for the keypoints and lines (in BGR format)
#     color_start = (0, 0, 255)  # Red
#     color_goal = (0, 255, 0)  # Green
#     color_path = (255, 0, 0)  # Blue
#     color_line = (255, 0, 255)  # White

#     # Draw lines and keypoints for start and goal configurations
#     for i in range(len(start_config) - 1):
#         cv2.line(image, tuple(start_config[i].astype(int)), tuple(start_config[i+1].astype(int)), color_line, 2)
#         cv2.line(image, tuple(goal_config[i].astype(int)), tuple(goal_config[i+1].astype(int)), color_line, 2)
#     for point in start_config:
#         cv2.circle(image, tuple(point.astype(int)), radius=5, color=color_start, thickness=-1)
#     for point in goal_config:
#         cv2.circle(image, tuple(point.astype(int)), radius=5, color=color_goal, thickness=-1)

#     # Draw lines and keypoints for intermediate configurations along the path
#     if path:
#         path_configs = [node.config for node in path]
#         for config in path_configs:
#             for i in range(len(config) - 1):
#                 cv2.line(image, tuple(config[i].astype(int)), tuple(config[i+1].astype(int)), color_line, 1)
#             for point in config:
#                 cv2.circle(image, tuple(point.astype(int)), radius=3, color=color_path, thickness=-1)

#     # Display the image
#     cv2.imshow('Path Visualization', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Optionally, save the image
#     save_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/path_with_keypoints.jpg'  # Modify as needed
#     cv2.imwrite(save_path, image)

def plot_path_on_image(image_path, path, start_config, goal_config):
    # Load the original image
    image = cv2.imread(image_path)

    # Function to generate a random color
    def random_color():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Draw start and goal keypoints (Red for start, Green for goal)
    for point in start_config:
        cv2.circle(image, tuple(point.astype(int)), radius=5, color=(0, 0, 255), thickness=-1)
    for point in goal_config:
        cv2.circle(image, tuple(point.astype(int)), radius=5, color=(0, 255, 0), thickness=-1)

    # Draw intermediate keypoints and lines for each configuration
    if path:
        path_configs = [node.config for node in path]
        for config in path_configs:
            color = random_color()  # Generate a random color for each configuration
            for i in range(len(config) - 1):
                cv2.line(image, tuple(config[i].astype(int)), tuple(config[i+1].astype(int)), color, 2)
            for point in config:
                cv2.circle(image, tuple(point.astype(int)), radius=3, color=color, thickness=-1)

    # Display and save the image
    # cv2.imshow('Path Visualization', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/path_with_keypoints_and_colored_lines.jpg', image)

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
    for idx, node in enumerate(path):
        image = base_image.copy()  # Copy the base image
        config = node.config
        for i in range(len(config) - 1):
            cv2.line(image, tuple(config[i].astype(int)), tuple(config[i+1].astype(int)), path_color(), 2)
        for point in config:
            cv2.circle(image, tuple(point.astype(int)), radius=3, color=path_color(), thickness=-1)
        
        # Save the image
        cv2.imwrite(os.path.join(output_directory, f'path_{idx}.jpg'), image)

# Main execution
if __name__ == "__main__":
    # Define the start and goal configurations (generalized for n keypoints)
    start_config = np.array([[258, 367], [258, 283], [180, 297], [176, 277], [175, 178], [198, 181]])  # Replace with your actual start configuration
    goal_config = np.array([[258, 367], [258, 283], [184, 254], [191, 235], [287, 212], [309, 217]])   # Replace with your actual goal configuration

    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/kprcnn_sim_latest/'  # Replace with the path to your JSON files
    configurations = load_keypoints_from_json(directory)

    # Detect the obstacle (green ball)
    image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/red_ball_image.jpg'  # Replace with the path to your image file
    obstacle_info = detect_red_ball(image_path)
    if obstacle_info is not None:
        obstacle_center, obstacle_radius = obstacle_info[:2], obstacle_info[2]
    else:
        print("No red ball detected in the image.")
        obstacle_center, obstacle_radius = None, None

    # Running the RRT algorithm
    # goal_node = modified_rrt(start_config, goal_config, configurations, obstacle_center, SAFE_DISTANCE)

    # Define a search radius for rewiring
    search_radius = 20  # Adjust as needed

    # Running the RRT* algorithm
    goal_node = rrt_star(start_config, goal_config, configurations, obstacle_center, SAFE_DISTANCE, search_radius)

    # path directory
    output_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/paths/path_4_rrt'

    # Extracting and plotting the path if found
    if goal_node:
        path = extract_path(goal_node)
        print("Path found:", path)
        plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)
    else:
        print("No path found")