#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2

# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
MAX_ITER = 500
STEP_SIZE = 10  # Pixels
SAFE_DISTANCE = 20  # Safe distance from the obstacle

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
# def is_too_close_to_obstacle(configuration, obstacle_center, safe_distance):
#     for point in configuration:
#         if np.linalg.norm(point - obstacle_center[:2]) < safe_distance:
#             return True
#     return False
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

def line_points(p1, p2):
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
    return points

def is_line_too_close_to_obstacle(p1, p2, obstacle_center, safe_distance):
    """Check if a line segment between p1 and p2 is too close to the obstacle."""
    for point in line_points(p1, p2):
        # print(point)
        # print(np.linalg.norm(np.array(point) - np.array(obstacle_center)))
        # print(np.array(point) - np.array(obstacle_center))
        # print(obstacle_center)
        if np.linalg.norm(np.array(point) - np.array(obstacle_center)) < safe_distance:
            return True
    return False

def is_configuration_too_close_to_obstacle(configuration, obstacle_center, safe_distance):
    """Check if any part of the robot configuration is too close to the obstacle."""
    for i in range(len(configuration) - 1):
        if is_line_too_close_to_obstacle(configuration[i], configuration[i+1], obstacle_center, safe_distance):
            return True
    return False


# def is_too_close_to_obstacle(configuration, obstacle_center, safe_distance):
#     for i in range(len(configuration) - 1):
#         p1 = configuration[i]
#         p2 = configuration[i + 1]
#         line_points = get_line_points(p1, p2)
#         for point in line_points:
#             if np.linalg.norm(np.array(point) - obstacle_center[:2]) < safe_distance:
#                 return True
#     return False

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
# def modified_rrt(start, goal, configurations, obstacle_center, safe_distance):
#     tree = [Node(start)]
#     for _ in range(MAX_ITER):
#         random_config = random.choice(configurations)  # Randomly choose a configuration from the dataset
#         nearest = nearest_node(tree, random_config)
#         new_config = steer(nearest.config, random_config, STEP_SIZE)
#         if not is_configuration_too_close_to_obstacle(new_config, obstacle_center, safe_distance):
#             new_node = Node(new_config, nearest)
#             tree.append(new_node)
#             if config_distance(new_config, goal) < STEP_SIZE:
#                 return new_node
#     return None

# Modified RRT algorithm
def modified_rrt(start, goal, configurations, obstacle_center, safe_distance):
    tree = [Node(start)]
    for _ in range(MAX_ITER):
        random_config = random.choice(configurations)
        nearest = nearest_node(tree, random_config)
        new_config = steer(nearest.config, random_config, STEP_SIZE)

        # Check each segment in the new configuration
        is_safe = True
        for i in range(len(new_config) - 1):
            if is_line_too_close_to_obstacle(new_config[i], new_config[i+1], obstacle_center, safe_distance):
                is_safe = False
                break

        if is_safe:
            new_node = Node(new_config, nearest)
            tree.append(new_node)
            if config_distance(new_config, goal) < STEP_SIZE:
                # Check the goal configuration
                if not is_configuration_too_close_to_obstacle(goal, obstacle_center, SAFE_DISTANCE):
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
    # goal_config = np.array([[258, 367], [258, 283], [184, 254], [191, 235], [287, 212], [309, 217]])   # Replace with your actual goal configuration
    goal_config = np.array([[257, 366], [257, 283], [303, 217], [320, 229], [403, 283], [389, 297]])

    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/kprcnn_sim_latest/'  # Replace with the path to your JSON files
    configurations = load_keypoints_from_json(directory)

    # Detect the obstacle (green ball)
    image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/red_ball_image_5.jpg'  # Replace with the path to your image file
    obstacle_info = detect_red_ball(image_path)
    if obstacle_info is not None:
        obstacle_center, obstacle_radius = obstacle_info[:2], obstacle_info[2]
    else:
        print("No red ball detected in the image.")
        obstacle_center, obstacle_radius = None, None

    # Running the RRT algorithm
    goal_node = modified_rrt(start_config, goal_config, configurations, obstacle_center, SAFE_DISTANCE)

    # path directory
    output_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/paths/path_7_rrt'

    # Extracting and plotting the path if found
    if goal_node:
        path = extract_path(goal_node)
        print("Path found:", path)
        plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)
    else:
        print("No path found")