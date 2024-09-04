#!/usr/bin/env python3
import json
import numpy as np
import cv2
import heapq
import os
import networkx as nx
import time
from sklearn.neighbors import KDTree, BallTree
# from scipy.spatial import KDTree 
import torchvision
from PIL import Image
import torch
import yaml
import shapely.geometry as geom
import scipy
import matplotlib.pyplot as plt
from pos_regression_control import PosRegModel
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import pickle, csv


# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

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
    # print("Shape of a single configuration:", configurations[0].shape)  
    return configurations

def load_keypoints_from_truncated_json(directory):
    configurations = []
    for filename in os.listdir(directory):
        # if filename.endswith('.json'):
        if filename.endswith('.json') and not filename.endswith('_joint_angles.json') and not filename.endswith('_vel.json'):
            file_index = int(filename.split('.')[0])
            if file_index >= 10000:
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    # Convert keypoints to integers
                    keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]  # Extracting x, y coordinates
                    configurations.append(np.array(keypoints))

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

def load_model_for_inference(model_path):    
    model = PosRegModel(18)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to inference mode
    return model

def predict_custom_distance(current_config, next_config, model):
    # Convert to 2D tensors if necessary
    start_kp_flat = torch.tensor(current_config.flatten(), dtype=torch.float).unsqueeze(0)  # Add batch dimension
    next_kp_flat = torch.tensor(next_config.flatten(), dtype=torch.float).unsqueeze(0)  # Add batch dimension

    # Predict the next configuration
    with torch.no_grad():
        output = model(start_kp_flat, next_kp_flat).squeeze(0).numpy()  # Remove batch dimension for output

    distance = np.linalg.norm(output)
    return float(distance)  # Reshape to the original configuration format

def custom_distance(x, y):
    # Ensure x and y are in the format the model expects (flattened arrays)
    return predict_custom_distance(x, y, model)

# Load the roadmap and KDTree from files
def load_graph_and_tree(graph_path, tree_path):
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    with open(tree_path, 'rb') as f:
        tree = pickle.load(f)
    print(f"Graph loaded from {graph_path}")
    print(f"KDTree loaded from {tree_path}")
    return graph, tree

def add_config_to_roadmap_no_obs(config, joint_angles, G, tree, k_neighbors):
    """Add a configuration to the roadmap, connecting it to its k nearest neighbors."""
    flattened_config = config.flatten().reshape(1, -1)
    _, indices = tree.query(flattened_config, k=k_neighbors)
    
    node_id = len(G.nodes)
    G.add_node(node_id, configuration=config, joint_angles=joint_angles)
    
    for i in indices[0]:
        G.add_edge(node_id, i)
    
    return node_id

# Function to add a configuration to the roadmap with collision checking
def add_config_to_roadmap_with_obs(config, joint_angles, G, tree, k_neighbors, obstacle_center, half_diagonal, safe_distance):
    # print("Shape of config being added:", config.shape)
    flattened_config = config.flatten().reshape(1, -1)
    _, indices = tree.query(flattened_config, k=k_neighbors)
    connections = 0    
    node_id = len(G.nodes)
    G.add_node(node_id, configuration=config, joint_angles=joint_angles)
    obstacle_boundary = geom.Polygon([
        (obstacle_center[0] - (half_diagonal + safe_distance), obstacle_center[1] - (half_diagonal + safe_distance)),
        (obstacle_center[0] + (half_diagonal + safe_distance), obstacle_center[1] - (half_diagonal + safe_distance)),
        (obstacle_center[0] + (half_diagonal + safe_distance), obstacle_center[1] + (half_diagonal + safe_distance)),
        (obstacle_center[0] - (half_diagonal + safe_distance), obstacle_center[1] + (half_diagonal + safe_distance)),
    ])
    
    for i in indices[0]:
        neighbor_config = G.nodes[i]['configuration']
        if is_collision_free(config, neighbor_config, obstacle_center, half_diagonal, safe_distance):
            G.add_edge(node_id, i)

    if nx.is_connected(G):
        print("Roadmap is connected")
    else:
        print("Roadmap is disconnected")  
    
    return node_id


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
                # If any segment intersects, the configuration is not collision-free
                return False
        
    for i in range(len(configuration1)):
        segment = geom.LineString([configuration1[i], configuration2[i]])
        if segment.intersects(obstacle_boundary):
            return False
        
     # If no segments intersect, the configuration is collision-free
    return True

def validate_and_remove_invalid_edges(G, obstacle_center, half_diagonal, safe_distance):
    # Iterate over a copy of the edges list to avoid modification issues during iteration
    for (u, v) in list(G.edges):
        config_u = G.nodes[u]['configuration']
        config_v = G.nodes[v]['configuration']
        # Perform the collision check for the edge
        if not is_collision_free(config_u, config_v, obstacle_center, half_diagonal, safe_distance):
            # If the edge is not collision-free, remove it from the graph
            G.remove_edge(u, v)
            # print(f"Removed invalid edge: {u} <-> {v}")
  
def find_path(G, start_node, goal_node):
    path_indices = nx.astar_path(G, source=start_node, target=goal_node)
    path_configurations = [[G.nodes[i]['configuration'], G.nodes[i]['joint_angles']] for i in path_indices]

    return path_configurations

def save_keypoints_and_joint_angles_to_csv(path, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Define headers
        headers = ['Config'] + [f'KP_{i}_x' for i in range(len(path[0][0]))] + [f'KP_{i}_y' for i in range(len(path[0][0]))] + ['Joint 1', 'Joint 2', 'Joint 3']
        csv_writer.writerow(headers)

        # Write each configuration and its joint angles
        for i, (config, angles) in enumerate(path):
            flat_config = [coord for kp in config for coord in kp]  # Flatten the keypoint configuration
            row = [f'Config_{i}'] + flat_config + list(angles)
            csv_writer.writerow(row)

def save_image_with_points(image_path, output_path, output_dir, points_sets):
    """
    Draws points and lines on an image and saves it to specified directories.

    Args:
    - image_path (str): Path to the input image.
    - output_path (str): Path to save the final modified image.
    - output_dir (str): Directory to save intermediate images.
    - points_sets (list): List of sets of points to draw.

    Returns:
    - None
    """
    
    # Load your image
    static_image = cv2.imread(image_path)
    
    if static_image is None:
        print(f"Error: Could not read the image at {image_path}.")
        return
    
    # Make a copy of the image for drawing
    gif_image = static_image.copy()

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define fixed colors for the points
    fixed_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    colors = np.random.randint(0, 255, (len(points_sets), 3))

    # Draw points and lines on the image
    for set_index, (points, color) in enumerate(zip(points_sets, colors)):
        if set_index == 0:
            for x, y in points:
                cv2.circle(static_image, (int(x), int(y)), 9, (0, 255, 0), -1)
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                cv2.line(static_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness=4)
        else:
            for index, (x, y) in enumerate(points):
                cv2.circle(static_image, (int(x), int(y)), 9, fixed_colors[index], -1)
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                cv2.line(static_image, (int(x1), int(y1)), (int(x2), int(y2)), tuple(int(c) for c in color), thickness=4)

        # Save the intermediate image with path number
        cv2.imwrite(os.path.join(output_dir, f'path_{set_index}.jpg'), static_image)

    # Save the final modified image
    cv2.imwrite(output_path, static_image)

def create_goal_image(goal_config, output_path):
    """
    Creates an image with specified keypoints and lines connecting them.

    Args:
    - goal_config (np.ndarray): Array of keypoints (x, y coordinates).
    - image_size (tuple): Size of the output image (height, width, channels).
    - circle_radius (int): Radius of the circles to draw.
    - circle_color (tuple): Color of the circles (BGR format).
    - line_color (tuple): Color of the lines (BGR format).
    - line_thickness (int): Thickness of the lines.
    - output_path (str): Path to save the output image.

    Returns:
    - None
    """
    # Initialize the blank image
    goal_image = np.zeros((480,640,3), dtype=np.int8)

    # Draw circles at each point in goal_config
    for point in goal_config:
        cv2.circle(goal_image, tuple(point.astype(int)), radius=9, color=(0, 0, 255), thickness=-1)

    # Draw lines between consecutive points in goal_config
    for i in range(len(goal_config) - 1):
        cv2.line(goal_image, tuple(goal_config[i].astype(int)), tuple(goal_config[i+1].astype(int)), (0, 0, 255), 4)

    # Save the image to the specified path
    cv2.imwrite(output_path, goal_image)
    print(f"Goal image successfully saved to {output_path}")

def draw_green_rectangle(image_path, rectangle_center, half_diagonal, save_path):
    # Read the image
    image = cv2.imread(image_path)

    # Calculate the full diagonal to get the rectangle width and height
    diagonal = 2 * half_diagonal
    width = int(diagonal / np.sqrt(2))
    height = width  # Assuming the rectangle is a square for simplicity

    # Calculate top left and bottom right points of the rectangle
    top_left = (rectangle_center[0] - width // 2, rectangle_center[1] - height // 2)
    bottom_right = (rectangle_center[0] + width // 2, rectangle_center[1] + height // 2)

    # Define the color of the rectangle (Green in BGR format)
    green_color = (0, 255, 0)

    # Draw the rectangle on the image
    cv2.rectangle(image, top_left, bottom_right, green_color, -1)

    # Save the new image
    cv2.imwrite(save_path, image)

def compute_obstacle_center(start_config, goal_config):
    """
    Computes the center of the rectangle formed by the last points of the start and goal configurations.

    Args:
    - start_config (np.ndarray): The start configuration, an array of keypoints (x, y coordinates).
    - goal_config (np.ndarray): The goal configuration, an array of keypoints (x, y coordinates).

    Returns:
    - obstacle_center (tuple): The (x, y) coordinates of the center of the rectangle.
    """
    # Extract the last point from each configuration
    start_point = start_config[-1]
    goal_point = goal_config[-1]

    # Compute the center of the rectangle formed by these two points
    obstacle_center = (int((start_point[0] + goal_point[0]) / 2), int((start_point[1] + goal_point[1]) / 2))

    return tuple(obstacle_center)

# Main execution
if __name__ == "__main__":
    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged/'  # Replace with the path to your JSON files
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b128_e500_v32.pth'
    configurations = load_keypoints_from_json(directory)
    model = load_model_for_inference(model_path)
    graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_roadmap_angle.pkl'
    tree_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom_tree_angle.pkl'
    file_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/'
    folder_num = 12

    # Define both folder paths
    exp_folder_no_obs = os.path.join(file_path, 'custom', 'no_obs', str(folder_num))
    exp_folder_with_obs = os.path.join(file_path, 'custom', 'with_obs', str(folder_num))

    # Create both folders if they don't exist
    for folder in [exp_folder_no_obs, exp_folder_with_obs]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # PRM parameters
    num_neighbors = 25

    # load roadmap for no collision check
    roadmap, tree = load_graph_and_tree(graph_path, tree_path)

    # Define start and goal configurations as numpy arrays
    start_config = np.array([[250, 442], [252, 311], [284, 261], [317, 210], [343, 226], [419, 219], [496, 212], [530, 207], [537, 250]])
    goal_config = np.array([[250, 442], [252, 311], [195, 297], [138, 283], [146, 253], [195, 198], [243, 143], [258, 114], [295, 132]])

    start_joint_angles = np.array([0.55919536054343, -1.14360361657187, 1.82161384974135])
    goal_joint_angles = np.array([-1.3369750023015, -2.27698595824979, 2.08266486041948])

    SAFE_ZONE = 40 
    obstacle_center = (400, 150)
    # obstacle_center = compute_obstacle_center(start_config, goal_config)
    print(obstacle_center)
    half_diagonal = 20

    start_node = add_config_to_roadmap_no_obs(start_config, start_joint_angles, roadmap, tree, num_neighbors)
    goal_node = add_config_to_roadmap_no_obs(goal_config, goal_joint_angles, roadmap, tree, num_neighbors)

    path_no_obs = find_path(roadmap, start_node, goal_node)

    save_keypoints_and_joint_angles_to_csv(path_no_obs, os.path.join(exp_folder_no_obs, 'joint_keypoints.csv'))

    if path_no_obs:
        point_set = []
        goal_sets = []
        # Iterate through the path, excluding the first and last configuration
        last_configuration = path_no_obs[-1][0]
        last_config = last_configuration[[3, 4, 6, 7, 8]]

        create_goal_image(last_config, os.path.join(exp_folder_no_obs, 'sim_published_goal_image_orig.jpg'))

        for configuration in path_no_obs[0:-1]:
            # Extract the last three keypoints of each configuration
            keypoints = configuration[0]
            selected_points = keypoints[[3, 4, 6, 7, 8]]
            selected_points_float = [[float(point[0]), float(point[1])] for point in selected_points]
            # Append these points to the point_set list
            point_set.append(selected_points_float)

        # Iterate through the path, excluding start and goal            
        for configuration in path_no_obs[1:]: 
            keypoints = configuration[0]
            selected_points = keypoints[[3, 4, 6, 7, 8]]
            selected_points_float = [[float(point[0]), float(point[1])] for point in selected_points]
            goal_features = []
            for point in selected_points_float:
                goal_features.extend(point)  # Add x, y as a pair
            goal_sets.append(goal_features)

        save_image_with_points(os.path.join(exp_folder_no_obs, 'sim_published_goal_image_orig.jpg'), \
                                os.path.join(exp_folder_no_obs, 'sim_published_goal_image_all.jpg'), \
                                    os.path.join(exp_folder_no_obs, 'path'), point_set)
         
        draw_green_rectangle(os.path.join(exp_folder_no_obs, 'sim_published_goal_image_all.jpg'), \
                              obstacle_center, half_diagonal, \
                                os.path.join(exp_folder_no_obs, 'sim_published_goal_image.jpg'))
           
        with open(os.path.join(exp_folder_no_obs, "dl_multi_features.yaml"), "w") as yaml_file:
            s = "dl_controller:\n"
            s += "  num_goal_sets: " + str(len(goal_sets)) + "\n"
            for i, goal in enumerate(goal_sets, start=1):
                # Convert the list of floats into a comma-separated string
                goal_str = ', '.join(map(str, goal))
                s += f"  goal_features{i}: [{goal_str}]\n"

            # Write the string to the file
            yaml_file.write(s)
    
        print("Data successfully written to dl_multi_features.yaml")

        # Save configurations to a .txt file
        
        with open(os.path.join(exp_folder_no_obs, "path_configurations_no_obs.txt"), "w") as file:
            file.write("Start Configuration:\n")
            file.write(str(start_config.tolist()) + "\n\n")
            file.write("Goal Configuration:\n")
            file.write(str(goal_config.tolist()) + "\n\n")
            file.write("Start Angle:\n")
            file.write(str(start_joint_angles.tolist()) + "\n\n")
            file.write("Goal Angle:\n")
            file.write(str(goal_joint_angles.tolist()) + "\n\n")
            file.write("Obstacle Parameters:\n")
            file.write("Safe Zone:\n")
            file.write(str(SAFE_ZONE) + "\n\n")
            file.write("Obstacle Center:\n")
            file.write(str(obstacle_center) + "\n\n")
            file.write("Half Diagonal:\n")
            file.write(str(half_diagonal) + "\n\n")
            file.write("Path:\n")
            for config, angles in path_no_obs:
                file.write(str(config.tolist()) + "\n")
            file.write("\nPoint Set:\n")
            for points in point_set:
                file.write(str(points) + "\n")

        print("Configurations successfully saved to configurations.txt")

    # load fresh roadmap for collision check
    roadmap, tree = load_graph_and_tree(graph_path, tree_path)

    obstacle_boundary = geom.Polygon([
        (obstacle_center[0] - (half_diagonal + SAFE_ZONE), obstacle_center[1] - (half_diagonal + SAFE_ZONE)),
        (obstacle_center[0] + (half_diagonal + SAFE_ZONE), obstacle_center[1] - (half_diagonal + SAFE_ZONE)),
        (obstacle_center[0] + (half_diagonal + SAFE_ZONE), obstacle_center[1] + (half_diagonal + SAFE_ZONE)),
        (obstacle_center[0] - (half_diagonal + SAFE_ZONE), obstacle_center[1] + (half_diagonal + SAFE_ZONE)), 
    ])


    # Add start and goal configurations to the roadmap
    start_node = add_config_to_roadmap_with_obs(start_config, start_joint_angles, roadmap, tree, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE)
    goal_node = add_config_to_roadmap_with_obs(goal_config, goal_joint_angles, roadmap, tree, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE) 

    validate_and_remove_invalid_edges(roadmap, obstacle_center, half_diagonal, SAFE_ZONE)
        
    # Find and print the path from start to goal
    path_with_obs = find_path(roadmap, start_node, goal_node)

    save_keypoints_and_joint_angles_to_csv(path_with_obs, os.path.join(file_path, os.path.join(exp_folder_with_obs, 'joint_keypoints.csv')))

    if path_with_obs:
        point_set = []
        goal_sets = []
        last_configuration = path_no_obs[-1][0]
        last_config = last_configuration[[3, 4, 6, 7, 8]]
        create_goal_image(last_config, os.path.join(exp_folder_with_obs, 'sim_published_goal_image_orig.jpg'))
        # Iterate through the path, excluding the first and last configuration
        for configuration in path_with_obs[0:-1]:
           # Extract the last three keypoints of each configuration
           keypoints = configuration[0]
           selected_points = keypoints[[3, 4, 6, 7, 8]]
           selected_points_float = [[float(point[0]), float(point[1])] for point in selected_points]
           # Append these points to the point_set list
           point_set.append(selected_points_float)
        # Iterate through the path, excluding start and goal            
        for configuration in path_with_obs[1:]: 
           keypoints = configuration[0]
           selected_points = keypoints[[3, 4, 6, 7, 8]]
           selected_points_float = [[float(point[0]), float(point[1])] for point in selected_points]
           goal_features = []
           for point in selected_points_float:
               goal_features.extend(point)  # Add x, y as a pair
           goal_sets.append(goal_features)
        save_image_with_points(os.path.join(exp_folder_with_obs, 'sim_published_goal_image_orig.jpg'), \
                            os.path.join(exp_folder_with_obs, 'sim_published_goal_image_all.jpg'), \
                                os.path.join(exp_folder_with_obs, 'path'), point_set)
        
        draw_green_rectangle(os.path.join(exp_folder_with_obs, 'sim_published_goal_image_all.jpg'), \
                          obstacle_center, half_diagonal, \
                            os.path.join(exp_folder_with_obs, 'sim_published_goal_image.jpg'))
        with open(os.path.join(exp_folder_with_obs, "dl_multi_features.yaml"), "w") as yaml_file:
            s = "dl_controller:\n"
            s += "  num_goal_sets: " + str(len(goal_sets)) + "\n"
            for i, goal in enumerate(goal_sets, start=1):
                # Convert the list of floats into a comma-separated string
                goal_str = ', '.join(map(str, goal))
                s += f"  goal_features{i}: [{goal_str}]\n"
            # Write the string to the file
            yaml_file.write(s)
        print("Data successfully written to dl_multi_features.yaml")
        # Save configurations to a .txt file
        with open(os.path.join(exp_folder_with_obs, "path_configurations_with_obs.txt"), "w") as file:
            file.write("Start Configuration:\n")
            file.write(str(start_config.tolist()) + "\n\n")
            file.write("Goal Configuration:\n")
            file.write(str(goal_config.tolist()) + "\n\n")
            file.write("Start Angle:\n")
            file.write(str(start_joint_angles.tolist()) + "\n\n")
            file.write("Goal Angle:\n")
            file.write(str(goal_joint_angles.tolist()) + "\n\n")
            file.write("Obstacle Parameters:\n")
            file.write("Safe Zone:\n")
            file.write(str(SAFE_ZONE) + "\n\n")
            file.write("Obstacle Center:\n")
            file.write(str(obstacle_center) + "\n\n")
            file.write("Half Diagonal:\n")
            file.write(str(half_diagonal) + "\n\n")
            file.write("Path:\n")
            for config, angles in path_with_obs:
                file.write(str(config.tolist()) + "\n")
            file.write("\nPoint Set:\n")
            for points in point_set:
                file.write(str(points) + "\n")
        print("Configurations successfully saved to configurations.txt")
    
    

