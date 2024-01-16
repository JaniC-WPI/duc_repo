#!/usr/bin/env python3

import json
import numpy as np
import cv2
import heapq
import os

# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
SAFE_DISTANCE = 10  # Safe distance from the obstacle

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
    print(points)
    return points

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

def is_configuration_too_close_to_obstacle(configuration, obstacle_center, safe_distance):
    """Check if any part of the robot configuration is too close to the obstacle."""
    for i in range(len(configuration) - 1):
        if is_line_too_close_to_obstacle(configuration[i], configuration[i+1], obstacle_center, safe_distance):
            return True
    return False

# Check if a configuration is too close to the obstacle
# def is_too_close_to_obstacle(configuration, obstacle_center, safe_distance):
#     for point in configuration:
#         if np.linalg.norm(point - obstacle_center[:2]) < safe_distance:
#             return True
#     return False

# def is_too_close_to_obstacle(configuration, obstacle_center, safe_distance):
#     for i in range(len(configuration) - 1):
#         p1 = configuration[i]
#         p2 = configuration[i + 1]
#         line_points = get_line_points(p1, p2)
#         for point in line_points:
#             if np.linalg.norm(np.array(point) - obstacle_center[:2]) < safe_distance:
#                 return True
#     return False

# Euclidean distance between configurations
def config_distance(config1, config2):
    return max(np.linalg.norm(p1 - p2) for p1, p2 in zip(config1, config2))

# def config_distance(config1, config2):
#     return sum(abs(p1 - p2).sum() for p1, p2 in zip(config1, config2))

# # A* search algorithm
# def a_star_search(start_idx, goal, config_dict, obstacle_center, safe_distance):
#     def heuristic(config_idx):
#         return np.linalg.norm(config_dict[config_idx] - goal)

#     open_set = [(0 + heuristic(start_idx), start_idx, [start_idx])]

#     while open_set:
#         current_cost, current_idx, path_indices = heapq.heappop(open_set)

#         if np.array_equal(config_dict[current_idx], goal):
#             return [config_dict[idx] for idx in path_indices]

#         for next_idx, next_config in config_dict.items():
#             if is_too_close_to_obstacle(next_config, obstacle_center, safe_distance):
#                 continue

#             next_cost = current_cost + config_distance(config_dict[current_idx], next_config) - heuristic(current_idx)
#             new_path_indices = path_indices + [next_idx]
#             heapq.heappush(open_set, (next_cost + heuristic(next_idx), next_idx, new_path_indices))

#     return None

def a_star_search_multiple_paths(start_config, goal_config, configurations, obstacle_center, safe_distance, num_paths):
    def heuristic(config):
        return np.linalg.norm(config - goal_config)

    def get_configuration_index(config, configurations):
        for i, conf in enumerate(configurations):
            if np.all(conf == config):
                return i
        return None

    start_index = get_configuration_index(start_config, configurations)
    if start_index is None:
        raise ValueError("Start configuration not found in the configurations.")

    open_set = [(0 + heuristic(start_config), start_index, [start_index])]
    paths = []

    while open_set and len(paths) < num_paths:
        current_cost, current_index, path_indices = heapq.heappop(open_set)
        current_config = configurations[current_index]

        if np.array_equal(current_config, goal_config):
            path = [configurations[i] for i in path_indices]
            if path not in paths:
                paths.append(path)
            continue

        for next_index, next_config in enumerate(configurations):
            if is_configuration_too_close_to_obstacle(next_config, obstacle_center, safe_distance):
                continue

            next_cost = current_cost + config_distance(current_config, next_config) - heuristic(current_config)
            new_path_indices = path_indices + [next_index]
            heapq.heappush(open_set, (next_cost + heuristic(next_config), next_index, new_path_indices))

    return paths

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
    for idx, config in enumerate(path):
        image = base_image.copy()  # Copy the base image
        for i in range(len(config) - 1):
            cv2.line(image, tuple(config[i].astype(int)), tuple(config[i+1].astype(int)), path_color(), 2)
        for point in config:
            cv2.circle(image, tuple(point.astype(int)), radius=3, color=path_color(), thickness=-1)
        
        # Save the image
        cv2.imwrite(os.path.join(output_directory, f'path_{idx}.jpg'), image)

def plot_paths(image_path, start_config, goal_config, paths, base_output_dir):
    for rank, path in enumerate(paths, start=1):
        output_dir = os.path.join(base_output_dir, f'rank_{rank}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)


# Main execution
if __name__ == "__main__":
    # Define the start and goal configurations (generalized for n keypoints)
    start_config = np.array([[257, 366], [257, 283], [179, 297], [175, 276], [175, 177], [197, 181]])  # Replace with your actual start configuration
    # goal_config = np.array([[257, 366], [257, 283], [303, 217], [320, 229], [403, 283], [389, 297]])   # Replace with your actual goal configuration
    goal_config = np.array([[257, 366], [257, 283], [183, 254], [191, 234], [287, 212], [309, 216]])

    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/kprcnn_sim_latest/'  # Replace with the path to your JSON files
    configurations = load_keypoints_from_json(directory)
    config_dict = {i: config for i, config in enumerate(configurations)}

    # Detect the obstacle (red ball)
    image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/red_ball_image_1_goal.jpg'  # Replace with the path to your image file
    obstacle_info = detect_red_ball(image_path)
    if obstacle_info is not None:
        obstacle_center, obstacle_radius = obstacle_info[:2], obstacle_info[2]
    else:
        print("No red ball detected in the image.")
        obstacle_center, obstacle_radius = None, None

    # Running the A* algorithm
    paths = a_star_search_multiple_paths(start_config, goal_config, configurations, obstacle_center, SAFE_DISTANCE, num_paths=2)
    if paths:
        for i, path in enumerate(paths, start=1):
            print(f"Path rank {i}:", path)
        plot_paths(image_path, start_config, goal_config, paths, '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/paths')
    else:
        print("No paths found")

    # path = a_star_search(0, goal_config, config_dict, obstacle_center, SAFE_DISTANCE)

    # path directory
    # output_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/paths/path_6_a*'

    # # Plotting the path if found
    # if path:
    #     print("Path found:", path)
    #     plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)
    # else:
    #     print("No path found")