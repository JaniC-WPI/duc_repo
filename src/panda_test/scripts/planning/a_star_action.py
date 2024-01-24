#!/usr/bin/env python3

import os
import json
import cv2
import heapq
import os
import numpy as np
import torch
from regression_model import KeypointRegressionNet

# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
SAFE_DISTANCE = 20  # Safe distance from the obstacle

def load_velocities_from_directory(directory):
    velocities = []

    # Iterate through each file in the directory
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('_vel.json'):
            filepath = os.path.join(directory, filename)

            # Open and load the JSON file
            with open(filepath, 'r') as file:
                data = json.load(file)

                # Extract the velocity and add it to the list
                if 'unique_velocity' in data:
                    velocities.append(data['unique_velocity'])

    return velocities

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
    return points

def is_line_too_close_to_obstacle(p1, p2, obstacle_center, safe_distance):
    """Check if a line segment between p1 and p2 is too close to the obstacle."""
    for point in line_points(p1, p2):
        if np.linalg.norm(np.array(point) - np.array(obstacle_center)) < safe_distance:
            return True
    return False

def is_configuration_too_close_to_obstacle(configuration, obstacle_center, safe_distance):
    """Check if any part of the robot configuration is too close to the obstacle."""
    for i in range(len(configuration) - 1):
        if is_line_too_close_to_obstacle(configuration[i], configuration[i+1], obstacle_center, safe_distance):
            return True
    return False

# load the regression model next keypoints detection        
def load_model_for_inference(model_path):
    # checkpoint = torch.load(model_path)
    # model = checkpoint['model_structure']
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()  # Set the model to inference mode
    # return model   
    model = KeypointRegressionNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to inference mode
    return model           

# Euclidean distance between configurations
def config_distance(config1, config2):
    return max(np.linalg.norm(p1 - p2) for p1, p2 in zip(config1, config2))

# Node class for creating the tree for A* search
class Node:
    def __init__(self, configuration, parent=None, g=0, h=0):
        self.configuration = configuration
        self.parent = parent
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic estimate to goal
        self.f = g + h  # Total cost

def heuristic(config1, config2):
    # Define your heuristic function here (e.g., Euclidean distance)
    return config_distance(config1, config2)

def a_star_search(start_config, goal_config, model_path, velocities, obstacle_center, obstacle_radius):
    # Load the regression model for inference
    model = load_model_for_inference(model_path)

    # Initialize start and end nodes
    start_node = Node(np.array(start_config), g=0, h=heuristic(start_config, goal_config))
    end_node = Node(np.array(goal_config))

    open_list = [start_node]
    closed_list = []

    while open_list:
        current_node = min(open_list, key=lambda node: node.f)
        open_list.remove(current_node)
        closed_list.append(current_node)

        # Check if goal is reached
        if np.array_equal(current_node.configuration, end_node.configuration):
            # Reconstruct path
            path = []
            while current_node is not None:
                path.append(current_node.configuration)
                current_node = current_node.parent
            return path[::-1]

        # Generate children
        for velocity in velocities:
            new_config = predict_next_configuration(current_node.configuration, velocity, model)
            print(new_config)
            
            # Check for collision with obstacle
            if obstacle_center and is_configuration_too_close_to_obstacle(new_config, obstacle_center, obstacle_radius + SAFE_DISTANCE):
                continue

            child_node = Node(new_config, parent=current_node, g=current_node.g + 1, h=heuristic(new_config, goal_config))

            # Check if child is in closed list
            if any(np.array_equal(child_node.configuration, closed_node.configuration) for closed_node in closed_list):
                continue

            # Check if child is in open list with lower g value
            if any(np.array_equal(child_node.configuration, open_node.configuration) and child_node.g > open_node.g for open_node in open_list):
                continue

            open_list.append(child_node)

    return None  # No path found

# Function to predict next configuration using the regression model
# def predict_next_configuration(current_config, velocity, model):
#     # Prepare the input for the model
#     start_kp_flat = torch.tensor(current_config.flatten(), dtype=torch.float)
#     velocity = torch.tensor(velocity, dtype=torch.float)
    
#     # Predict the next configuration
#     with torch.no_grad():
#         next_kp_flat = model(start_kp_flat, velocity).numpy()
#     return next_kp_flat.reshape(-1, 2)  # Reshape to the original configuration format

def predict_next_configuration(current_config, velocity, model):
    # Convert to 2D tensors if necessary
    start_kp_flat = torch.tensor(current_config.flatten(), dtype=torch.float).unsqueeze(0)  # Add batch dimension
    velocity = torch.tensor(velocity, dtype=torch.float).unsqueeze(0)  # Add batch dimension

    # Predict the next configuration
    with torch.no_grad():
        next_kp_flat = model(start_kp_flat, velocity).squeeze(0).numpy()  # Remove batch dimension for output

    new_config = next_kp_flat.reshape(-1, 2)
    return new_config.astype(int)  # Reshape to the original configuration format



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

if __name__ == '__main__':
    # Define the start and goal configurations (generalized for n keypoints)
    start_config = np.array([[257, 366], [257, 283], [179, 297], [175, 276], [175, 177], [197, 181]])  # Replace with your actual start configuration
    # goal_config = np.array([[257, 366], [257, 283], [303, 217], [320, 229], [403, 283], [389, 297]])   # Replace with your actual goal configuration
    goal_config = np.array([[257, 366], [257, 283], [183, 254], [191, 234], [287, 212], [309, 216]])

    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/vel_reg_sim_test_2/unique_vel/'
    velocities = load_velocities_from_directory(directory)
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_nkp_b128_e300_v1.pth'

    # Detect the obstacle (red ball)
    image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/red_ball_image_1_goal.jpg'  # Replace with the path to your image file
    obstacle_info = detect_red_ball(image_path)
    if obstacle_info is not None:
        obstacle_center, obstacle_radius = obstacle_info[:2], obstacle_info[2]
    else:
        print("No red ball detected in the image.")
        obstacle_center, obstacle_radius = None, None

    path = a_star_search(start_config, goal_config, model_path, velocities, obstacle_center, obstacle_radius)

    # path directory
    output_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/paths/paths_astar/path_1_a*'

    # Plotting the path if found
    if path:
        print("Path found:", path)
        plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)
    else:
        print("No path found")

