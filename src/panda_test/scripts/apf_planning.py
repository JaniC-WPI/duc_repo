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

def line_points(p1, p2, interval=20):
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

def total_attractive_potential(current_config, goal_config, alpha):
    total_force = np.zeros_like(current_config)
    for i, current_position in enumerate(current_config):
        goal_position = goal_config[i]
        total_force[i] = alpha * (current_position - goal_position)
    return total_force

def compute_repulsive_force_for_line_segment(p1, p2, obstacle_position, beta, safe_distance):
    line_pts = line_points(p1, p2)
    total_force = np.zeros(2)
    for pt in line_pts:
        distance = np.linalg.norm(np.array(pt) - np.array(obstacle_position))
        if distance < safe_distance:
            force = beta * (1/distance - 1/safe_distance) * (np.array(pt) - obstacle_position) / distance**2
            total_force += force
    return total_force

def total_repulsive_potential(current_config, obstacle_position, beta, safe_distance):
    total_force = np.zeros_like(current_config)
    for i in range(len(current_config) - 1):
        p1 = current_config[i]
        p2 = current_config[i + 1]
        total_force[i] += compute_repulsive_force_for_line_segment(p1, p2, obstacle_position, beta, safe_distance)
    return total_force

def compute_motion(start_config, goal_config, obstacle_positions, alpha, beta, safe_distance, max_iterations, step_size):
    path = [start_config]
    current_config = np.array(start_config, dtype=float)

    for _ in range(max_iterations):
        attractive_force = total_attractive_potential(current_config, np.array(goal_config), alpha)
        repulsive_force = np.zeros_like(current_config)
        for obstacle in obstacle_positions:
            repulsive_force += total_repulsive_potential(current_config, np.array(obstacle), beta, safe_distance)

        net_force = attractive_force + repulsive_force

        # Normalize and apply step size
        for i in range(len(current_config)):
            if np.linalg.norm(net_force[i]) > 0:
                current_config[i] += (net_force[i] / np.linalg.norm(net_force[i])) * step_size

        path.append(current_config.tolist())

        # Check if goal is reached
        if np.linalg.norm(current_config - np.array(goal_config)) < step_size:
            break

    return path

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

    # Define APF parameters
    alpha = 1.0  # Coefficient for attractive potential
    beta = 10.0  # Coefficient for repulsive potential
    safe_distance = 50  # Distance for repulsive potential to act
    max_iterations = 10
    step_size = 50

    # Define the start and goal configurations (generalized for n keypoints)
    start_config = np.array([[257, 366], [257, 283], [179, 297], [175, 276], [175, 177], [197, 181]])  # Replace with your actual start configuration
    # goal_config = np.array([[257, 366], [257, 283], [303, 217], [320, 229], [403, 283], [389, 297]])   # Replace with your actual goal configuration
    goal_config = np.array([[257, 366], [257, 283], [183, 254], [191, 234], [287, 212], [309, 216]])

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

    path = compute_motion(start_config, goal_config, obstacle_center, alpha, beta, safe_distance, max_iterations, step_size)    

    # path directory
    output_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/paths/path_1_apf'

    # Plotting the path if found
    if path:
        print("Path found:", path)
        plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)
    else:
        print("No path found")





