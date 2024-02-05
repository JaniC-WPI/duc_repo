#!/usr/bin/env python3

import os
import json
import cv2
import heapq
import os
import numpy as np
import torch
from regression_path_planning import KeypointRegressionNet
import time
import csv

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

# def line_points(p1, p2):
#     """Return all the points on the line segment from p1 to p2 using Bresenham's Line Algorithm."""
#     x1, y1 = p1
#     x2, y2 = p2
#     points = []
#     dx = abs(x2 - x1)
#     dy = abs(y2 - y1)
#     x, y = x1, y1
#     sx = 1 if x1 < x2 else -1
#     sy = 1 if y1 < y2 else -1
#     if dx > dy:
#         err = dx / 2.0
#         while x != x2:
#             points.append((x, y))
#             err -= dy
#             if err < 0:
#                 y += sy
#                 err += dx
#             x += sx
#     else:
#         err = dy / 2.0
#         while y != y2:
#             points.append((x, y))
#             err -= dx
#             if err < 0:
#                 x += sx
#                 err += dy
#             y += sy
#     points.append((x, y))
#     return points

# def is_line_too_close_to_obstacle(p1, p2, obstacle_center, safe_distance):
#     """Check if a line segment between p1 and p2 is too close to the obstacle."""
#     for point in line_points(p1, p2):
#         if np.linalg.norm(np.array(point) - np.array(obstacle_center)) < safe_distance:
#             return True
#     return False

def is_configuration_too_close_to_obstacle(configuration, obstacle_center, safe_distance):
    """Check if any part of the robot configuration is too close to the obstacle."""
    for i in range(len(configuration) - 1):
        if is_line_too_close_to_obstacle(configuration[i], configuration[i+1], obstacle_center, safe_distance):
            return True
    return False

def point_to_line_distance(point, line_start, line_end):
    """Calculate the distance from a point to a line segment."""
    # Line segment vector
    line_vec = line_end - line_start
    # Vector from line start to the point
    point_vec = point - line_start

    # Calculate the projection of the point vector onto the line segment vector
    line_len = np.linalg.norm(line_vec)
    projection = np.dot(point_vec, line_vec) / line_len**2

    # Check where the projection lies
    if projection < 0.0:
        closest_point = line_start
    elif projection > 1.0:
        closest_point = line_end
    else:
        closest_point = line_start + projection * line_vec

    # Distance from the closest point on the line segment to the point
    return np.linalg.norm(closest_point - point)

def is_line_too_close_to_obstacle(p1, p2, obstacle_center, safe_distance):
    """Check if a line segment between p1 and p2 is too close to the obstacle."""
    p1 = np.array(p1)
    p2 = np.array(p2)
    obstacle_center = np.array(obstacle_center)

    distance = point_to_line_distance(obstacle_center, p1, p2)
    return distance <= safe_distance


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
    distances = [np.linalg.norm(p1 - p2) for p1, p2 in zip(config1, config2)]
    # print("distances", distances)
    avg_cost = sum(distances)/len(distances)
    return avg_cost
    # return max(np.linalg.norm(p1 - p2) for p1, p2 in zip(config1, config2))

# Node class for creating the tree for A* search
class Node:
    def __init__(self, configuration, parent=None):
        self.configuration = configuration
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return np.array_equal(self.configuration, other.configuration)
    
    # def is_similar(self, other_node, threshold=2):
    #     """Check if the configurations of two nodes are similar within a given threshold."""
    #     diff = np.abs(self.configuration - other_node.configuration)
    #     # Check if all x and y values of keypoints are within the threshold
    #     return np.all(diff <= threshold)

    def is_similar(self, other_node, threshold=4):
        """Check if the overall configurations are similar within a given threshold."""
        # Calculate the Euclidean distance between corresponding keypoints
        distance = np.linalg.norm(self.configuration - other_node.configuration)
        print("distance threshold", distance)
        return distance <= threshold

def heuristic(config1, config2):
    # Define your heuristic function here (e.g., Euclidean distance)
    # return np.linalg.norm(config1 - config2)
    return config_distance(config1, config2)

def a_star_search(start_config, goal_config, model_path, velocities, obstacle_center, obstacle_radius):
    # Load the regression model for inference
    model = load_model_for_inference(model_path)

    # Initialize start and end nodes
    start_node = Node(np.array(start_config)) #g=0, h=heuristic(start_config, goal_config))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(np.array(goal_config))
    end_node.g = end_node.h = end_node.f = 0

    open_list = [start_node]
    closed_list = []

    # Open a CSV file to write the data
    with open('/home/jc-merlab/Pictures/panda_data/panda_sim_vel/csv/path_predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'Velocity', 'Input Config', 'Predicted Config'])  # Header

        iteration = 0
        max_iterations = 1000

        while len(open_list) > 0 and iteration < max_iterations:
            # current_node = min(open_list, key=lambda node: node.f)
            current_node =  open_list[0]
            current_index =  0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index
    
            open_list.pop(current_index)
            closed_list.append(current_node)
    
            # Check if goal is reached
            if current_node.is_similar(end_node):
                path = []
                current = current_node
                while current is not None:
                    path.append(current.configuration)
                    current = current.parent
                return path[::-1]
    
            children = []
            # Generate children
            for velocity in velocities:
                # print("Open List", [node.configuration for node in open_list])
                # print("Closed List", [node.configuration for node in closed_list])
                # print("Current Node", current_node.configuration)
                new_config = predict_next_configuration(current_node.configuration, velocity, model)            
                # print("for veloctiy", velocity, "New Config", new_config)

                # Write iteration, velocity, and new configuration to CSV
                writer.writerow([iteration, velocity, current_node.configuration.tolist(), new_config.tolist()])
                
                # Check for collision with obstacle
                if obstacle_center and is_configuration_too_close_to_obstacle(new_config, obstacle_center, obstacle_radius + SAFE_DISTANCE):
                    print("configuration too close")
                    continue
                else:
                    print("clear for planning")
                
                distance = config_distance(current_node.configuration, new_config)
    
                new_node =  Node(new_config, current_node)
    
                children.append(new_node)
    
            # print("Children nodes", [node.configuration for node in children])
    
            for child_node in children:
                # Check if child is in closed list or similar to any node in the closed list
                if any(child_node.is_similar(closed_node) for closed_node in closed_list):
                    print("child is in closed list")
                    continue
                else:
                    print("Child is not in Closed List")
                
                child_node.g = current_node.g + distance
                child_node.h = heuristic(child_node.configuration, goal_config)
                child_node.f = child_node.g + child_node.h
    
                # # Check if similar node is already in open list
                # if any(child_node.is_similar(open_node) and child_node.g > open_node.g for open_node in open_list):
                #     print("child is in open list but with higher cost")
                #     continue
                # else:
                #     print("Child is in open list with lower cost")
                
                open_list.append(child_node)
            
            iteration += 1

    if iteration >= max_iterations:
        print("Reached maximum iterations without finding a path.")

    return None  # No path found

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
    # start_config = np.array([[257, 366], [257, 283], [179, 297], [175, 276], [175, 177], [197, 181]])  # Replace with your actual start configuration
    # goal_config = np.array([[257, 366], [257, 283], [303, 217], [320, 229], [403, 283], [389, 297]])   # Replace with your actual goal configuration
    # goal_config = np.array([[257, 366], [257, 283], [183, 254], [191, 234], [287, 212], [309, 216]])
    start_config = np.array([[291.3727, 425.9008], [292.9618, 305.4389], [186.7145, 290.6239], [190.7097,
        262.4118], [240.0366, 390.2596], [211.3978, 398.0443]])
    
    goal_config = np.array([[291.3727, 425.9008], [292.9618, 305.4389], [185.5897, 311.7277], [184.3402,
        283.1863], [317.4286, 268.2987], [312.1099, 292.2249]])

    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/unique_vel_planning/1/'
    velocities = load_velocities_from_directory(directory)
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_nkp_b64_e500_v1_l5.pth'

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

