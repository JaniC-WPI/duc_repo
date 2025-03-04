#!/usr/bin/env python3

import json
import numpy as np
import cv2
import heapq
import os
import networkx as nx
import time
from sklearn.neighbors import KDTree
from descartes import PolygonPatch
from sklearn.neighbors import KDTree, BallTree
import shapely.geometry as geom
import torch
from scripts.planning.control.pos_regression_control import PosRegModel

# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
SAFE_DISTANCE = 20  # Safe distance from the obstacle

# Load keypoints from JSON files in a given directory
def load_keypoints_from_json(directory):
    configurations = []
    for filename in os.listdir(directory):
        # if filename.endswith('.json'):
        if filename.endswith('.json') and not filename.endswith('_joint_angles.json') and not filename.endswith('_vel.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                # Convert keypoints to integers
                keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]  # Extracting x, y coordinates
                configurations.append(np.array(keypoints))
    # print(configurations)
    return configurations

def load_model_for_inference(model_path):    
    model = PosRegModel(12)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to inference mode
    return model

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

def is_collision_free(configuration1, configuration2, obstacle_center, half_diagonal, safe_distance):
    # Define the square boundary of the obstacle including the safe distance
    obstacle_boundary = geom.Polygon([
        (obstacle_center[0] - (half_diagonal + safe_distance), obstacle_center[1] - (half_diagonal + safe_distance)),
        (obstacle_center[0] + (half_diagonal + safe_distance), obstacle_center[1] - (half_diagonal + safe_distance)),
        (obstacle_center[0] + (half_diagonal + safe_distance), obstacle_center[1] + (half_diagonal + safe_distance)),
        (obstacle_center[0] - (half_diagonal + safe_distance), obstacle_center[1] + (half_diagonal + safe_distance)),
    ])

    # Check for collision between consecutive keypoints within the same configuration
    for config in [configuration1, configuration2]:
        for i in range(len(config) - 1):
            segment = geom.LineString([config[i], config[i+1]])
            if segment.intersects(obstacle_boundary):
                print("collision detected")
                # If any segment intersects, the configuration is not collision-free
                return False
        
    for i in range(len(configuration1)):
        segment = geom.LineString([configuration1[i], configuration2[i]])
        if segment.intersects(obstacle_boundary):
            print("edge collision detected")
            return False
        
     # If no segments intersect, the configuration is collision-free
    return True

def predict_custom_distance(current_config, next_config, model):
    # Convert to 2D tensors if necessary
    start_kp_flat = torch.tensor(current_config.flatten(), dtype=torch.float).unsqueeze(0)  # Add batch dimension
    next_kp_flat = torch.tensor(next_config.flatten(), dtype=torch.float).unsqueeze(0)  # Add batch dimension

    # Predict the next configuration
    with torch.no_grad():
        output = model(start_kp_flat, next_kp_flat).squeeze(0).numpy()  # Remove batch dimension for output

    distance = np.linalg.norm(output)
    return float(distance)  # Reshape to the original configuration format

def build_roadmap_kd_tree(configurations, start_config, goal_config, k, obstacle_center, half_diagonal, safe_distance):
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
    # tree = KDTree(all_configs_np)
    tree = BallTree(all_configs_np, metric=lambda x, y: predict_custom_distance(x, y, model))

    for i, config in enumerate(all_configs):
         # Reshape the config to be 2D array for KDTree query
        config_reshaped = config.flatten().reshape(1, -1)
        # Query the k nearest neighbors
        distances, indices = tree.query(config_reshaped, k=k+1)  # k+1 because the query point itself is included

        for j in range(1, k+1):  # Skip the first index (itself)
            neighbor_config = all_configs[indices[0][j]]
            if is_collision_free(config, neighbor_config, obstacle_center, half_diagonal, safe_distance):
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
    start_config = np.array([[267, 432], [269, 315], [200, 237], [219, 217], [322, 139], [344, 115]]) 
    goal_config = np.array([[267, 431], [269, 315], [240, 213], [266, 206], [387, 256], [417, 243]])
    # goal_config = np.array([[258, 367], [258, 282], [303, 217], [320, 229], [403, 283], [389, 297]])  
    # goal_config = np.array([[257, 366], [257, 283], [183, 254], [191, 234], [287, 212], [309, 216]])

    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_planning_kprcnn/'  # Replace with the path to your JSON files
    configurations = load_keypoints_from_json(directory)
    configurations = configurations[1:9000:10]

    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b64_e400_v6.pth'
    num_samples = 500
    configurations = load_keypoints_from_json(directory)
    model = load_model_for_inference(model_path)

    # Load and sample configurations from JSON files
    num_samples = 500 # Adjust as needed
    # configurations = load_and_sample_configurations(directory, num_samples)


    # Detect the obstacle (red ball)
    # image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/red_ball_image_1_goal.jpg'  # Replace with the path to your image file
    # obstacle_info = detect_red_ball(image_path)
    # if obstacle_info is not None:
    #     obstacle_center, obstacle_radius = obstacle_info[:2], obstacle_info[2]
    # else:
    #     print("No red ball detected in the image.")
    #     obstacle_center, obstacle_radius = None, None

    SAFE_ZONE = 50  # Safe distance from the obstacle
    obstacle_center = (420, 133)
    half_diagonal = 20

    # Parameters for PRM
    num_neighbors = 25  # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    roadmap = build_roadmap_kd_tree(configurations, start_config, goal_config, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE)
    end_time = time.time()

    print("time taken to plan path", end_time - start_time)

    # Find the path
    path = find_path_prm(roadmap, start_config, goal_config)

    # path directory
    # output_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/paths/path_prm/path_12_prm'

    # Plotting the path if found
    # if path:
    #     print("Path found:", path)
    #     # plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)
    # else:
    #     print("No path found")

    # end_time = time.time()

    if path:
        point_set = []
        goal_sets = []
        # Iterate through the path, excluding the first and last configuration
        for configuration in path[1:-1]:
            # Extract the last three keypoints of each configuration
            last_three_points = configuration[-4:]
            last_three_points_float = [[float(point[0]), float(point[1])] for point in last_three_points]
            # Append these points to the point_set list
            point_set.append(last_three_points_float)
        # Iterate through the path, excluding start and goal
        for configuration in path[1:]: 
            last_three_points = configuration[-4:]
            last_three_points_float = [[float(point[0]), float(point[1])] for point in last_three_points]
            goal_features = []  # Create a new list for each goal set
            for point in last_three_points_float:
                goal_features.extend(point)  # Add x, y as a pair
            goal_sets.append(goal_features)
        print("Point Set:", point_set)
        print("goal sets: ", goal_sets)

        with open("config/dl_multi_features.yaml", "w") as yaml_file:
            s = "dl_controller:\n"
            s += "  num_goal_sets: " + str(len(goal_sets)) + "\n"
            for i, goal in enumerate(goal_sets, start=1):
                # Convert the list of floats into a comma-separated string
                goal_str = ', '.join(map(str, goal))
                s += f"  goal_features{i}: [{goal_str}]\n"

            # Write the string to the file
            yaml_file.write(s)

        print("Data successfully written to config/dl_multi_features.yaml")


