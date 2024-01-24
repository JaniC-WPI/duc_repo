#!/usr/bin/env python3

import json
import numpy as np
import cv2
import heapq
import os
import torch

SAFE_DISTANCE = 20

# load transition data from a direcoty in device
def load_transition_data(directory):
    transitions = {}
    for filename in os.listdir(directory):
        if filename.endswith('_combined.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)

                # Extract the x, y coordinates and convert to tuples
                start_kp = tuple(tuple(kp[0][:2] for kp in data['start_kp']))
                next_kp = tuple(tuple(kp[0][:2] for kp in data['next_kp']))
                velocity = data['velocity']  # Keep velocity as floating-point

                if start_kp not in transitions:
                    transitions[start_kp] = []
                transitions[start_kp].append((next_kp, velocity))
    return transitions

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

# Heuristic function for the A* algorithm
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star(start, goal, transitions, obstacle_center, safe_distance):
    # Convert start and goal configurations to match the format in transitions
    start = tuple(tuple(p[:2]) for p in start)
    goal = tuple(tuple(p[:2]) for p in goal)

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        # if current == goal:  # Check if goal is reached
        #     return reconstruct_path(came_from, current)

        # Check if goal is "close enough" to the current position
        if np.allclose(np.array(current), np.array(goal), atol=1e-6):
            return reconstruct_path(came_from, current)

        for next_kp, _ in transitions.get(current, []):
            next_kp = tuple(tuple(p) for p in next_kp)
            tentative_g_score = g_score[current] + heuristic(current, next_kp)

            if next_kp not in g_score or tentative_g_score < g_score[next_kp]:
                came_from[next_kp] = current
                g_score[next_kp] = tentative_g_score
                f_score[next_kp] = tentative_g_score + heuristic(next_kp, goal)
                if next_kp not in open_set and is_collision_free(np.array(next_kp), obstacle_center, safe_distance):
                    heapq.heappush(open_set, (f_score[next_kp], next_kp))

    return None

def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

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
    start_config = np.array([[257.95220042652915, 366.9198630617724], [257.95973939799904, 283.013113744617], 
                             [179.53457014392896, 297.13509063783573], [175.87658469885523, 276.66301779791337], 
                             [175.30964903682178, 177.73791800590934], [197.5608564429075, 181.2016105905464]])  
    # goal_config = np.array([[257.95220042652915, 366.9198630617724], [257.95973939799904, 283.013113744617], 
    #                         [303.0927220551031, 217.43165270791394], [320.21581019993806, 229.21361657394115],  
    #                         [403.5322106093911, 283.0647464931091], [389.2718701618588, 297.7476490616017]])
    goal_config = np.array([[257.95220042652915, 366.9198630617724], [257.95973939799904, 283.013113744617], 
                            [183.54445371948276, 254.32856622097543], [191.0549620237761, 234.91962096219095], 
                            [287.4736733202998, 212.0599437741952], [309.0946027096692, 216.72953260678656]])

    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/vel_reg_sim_test/'  # Replace with the path to your JSON files
    # configurations = load_keypoints_from_json(directory)
    transitions = load_transition_data(directory)

    # Detect the obstacle (red ball)
    image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/red_ball_image_1_goal.jpg'  # Replace with the path to your image file
    obstacle_info = detect_red_ball(image_path)
    if obstacle_info is not None:
        obstacle_center, obstacle_radius = obstacle_info[:2], obstacle_info[2]
    else:
        print("No red ball detected in the image.")
        obstacle_center, obstacle_radius = None, None

    # Run the path planning
    path = a_star(start_config, goal_config, transitions, obstacle_center, SAFE_DISTANCE)

    # path directory
    output_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/paths/path_1_tran_a_astar'

    # Plotting the path if found
    if path:
        print("Path found:", path)
        plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)
    else:
        print("No path found")

