#!/usr/bin/env python3
import json
import numpy as np
import cv2
import heapq
import os
import networkx as nx
import time
from sklearn.neighbors import KDTree
import torchvision
from PIL import Image
import torchvision.transforms as T
import yaml
import shapely.geometry as geom

# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
SAFE_DISTANCE = 50  # Safe distance from the obstacle
# COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Load keypoints from JSON files in a given directory
def load_keypoints_from_json(directory):
    configurations = []
    for filename in os.listdir(directory):
        # if filename.endswith('.json'):
        if filename.endswith('.json') and not filename.endswith('_combined.json') and not filename.endswith('_vel.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                # Convert keypoints to integers
                keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]  # Extracting x, y coordinates
                configurations.append(np.array(keypoints))
    # print(configurations)
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

def detect_green_rectangle(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range for green color
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate half of the diagonal using Pythagoras theorem
        half_diagonal = np.sqrt(w**2 + h**2) / 2
        return (int(x + w/2), int(y + h/2), int(half_diagonal))
    return None

def build_lazy_roadmap_with_kdtree(configurations, k_neighbors):
    """
    Build a LazyPRM roadmap using a KDTree for efficient nearest neighbor search.
    
    Args:
    - configurations: List[np.array], a list of configurations (keypoints flattened).
    - k_neighbors: int, the number of neighbors to connect to each node.
    
    Returns:
    - G: nx.Graph, the constructed roadmap.
    """
    flattened_configs = np.vstack([config.flatten() for config in configurations])
    tree = KDTree(flattened_configs)
    G = nx.Graph()
    
    for i, config in enumerate(configurations):
        G.add_node(i, configuration=config)
        
    for i, config in enumerate(flattened_configs):
        _, indices = tree.query([config], k=k_neighbors + 1)  # +1 to include self in results
        for j in indices[0][1:]:  # Skip self
            G.add_edge(i, j)
            
    return G, tree

# Function to add a configuration to the roadmap with collision checking
def add_config_to_roadmap(config, G, tree, k_neighbors, obstacle_center, safe_distance, half_diagonal):
    flattened_config = config.flatten().reshape(1, -1)
    _, indices = tree.query(flattened_config, k=k_neighbors)
    connections = 0
    
    node_id = len(G.nodes)
    G.add_node(node_id, configuration=config)
    
    for i in indices[0]:
        neighbor_config = G.nodes[i]['configuration']
        # Here we need to convert configurations back to their original shape for collision checking
        if is_collision_free(np.vstack((config, neighbor_config)), obstacle_center, safe_distance, half_diagonal):
            G.add_edge(node_id, i)
            connections += 1

    if connections == 0:  # If no connections were made, remove the node
        G.remove_node(node_id)
        return None, tree
    
    # Update the tree with the new node for future queries
    new_flattened_configs = np.vstack([tree.data, flattened_config])
    new_tree = KDTree(new_flattened_configs)
    
    return node_id, new_tree

def distance_line_to_point(p1, p2, point):
    """Calculates the distance of a point to a line segment."""
    numerator = np.abs((p2[1] - p1[1]) * point[0] - (p2[0] - p1[0]) * point[1] + p2[0] * p1[1] - p2[1] * p1[0])
    denominator = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    return numerator / denominator

def square_obstacle(center, half_diagonal):
    """Creates a Shapely square polygon representing the obstacle."""
    dx = dy = half_diagonal
    x0, y0 = center[0] - dx, center[1] - dy  # Bottom-left corner
    return geom.Polygon(((x0, y0), (x0 + 2*dx, y0), (x0 + 2*dx, y0 + 2*dy), (x0, y0 + 2*dy)))

def is_collision_free(configuration, obstacle_center, safe_distance, half_diagonal):
    obstacle = square_obstacle(obstacle_center, half_diagonal + safe_distance)

    for i in range(len(configuration) - 1):
        line_segment = geom.LineString([configuration[i], configuration[i + 1]])
        if line_segment.distance(obstacle) <= 0:  # Collision! 
            return False

    return True  # Collision-free

def find_path(G, start_node, goal_node):
    """Find a path from start to goal in the roadmap G."""
    path_indices = nx.astar_path(G, source=start_node, target=goal_node)
    path_configurations = [G.nodes[i]['configuration'] for i in path_indices]
    return path_configurations

# Main execution
if __name__ == "__main__":
    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/path_planning_panda_regression/'  # Replace with the path to your JSON files
    num_samples = 500
    configurations = load_keypoints_from_json(directory)
    # configurations = load_and_sample_configurations(directory, num_samples)
    # Parameters for PRM
    num_neighbors = 500  # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    roadmap, tree = build_lazy_roadmap_with_kdtree(configurations, num_neighbors)   
    end_time = time.time()

    print("time taken to find the graph", end_time - start_time)  

    # Define start and goal configurations as numpy arrays
    start_config = np.array([[271, 431], [270, 313], [194, 240], [214, 221], [300, 124], [312, 95]])
    goal_config = np.array([[271, 431], [271, 313], [243, 211], [270, 203], [389, 258], [418, 243]])
    
    obstacle_center = (380, 133)
    half_diagonal = 20
    safe_distance = half_diagonal + SAFE_DISTANCE 

    # Add start and goal configurations to the roadmap
    start_node, tree = add_config_to_roadmap(start_config, roadmap, tree, num_neighbors, obstacle_center, safe_distance, half_diagonal)
    goal_node, tree = add_config_to_roadmap(goal_config, roadmap, tree, num_neighbors, obstacle_center, safe_distance, half_diagonal)
        
    # Find and print the path from start to goal
    path = find_path(roadmap, start_node, goal_node)
    print("Path from start to goal:", path)

    # After finding and printing the path from start to goal
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

            # # Append these points to the point_set list
            # point_set.append(last_three_points.tolist())  # Convert to list if necessary

        print("Point Set:", point_set)
        print("goal sets: ", goal_sets)

        # Prepare the data for the YAML file
        # Prepare the data for the YAML file
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
        # s += "  goal_features: [" + (','.join(map(str,srv_resp.kp.data))) +"]\n"

        # for i, goal_set in enumerate(point_set):
        #     yaml_data['dl_controller'][f'goal_features{i+1}'] = goal_set

        # Write to a YAML file
        # yaml_file_path = 'config/dl_multi_features.yaml'  # Specify the path to your YAML file




    

    # image_path = '/home/jc-merlab/Pictures/panda_data/images_for_occlusion/1/image_864.jpg'
    # # results = object_detection_yolov8(image_path)
    # # results = object_detection_ycb_roboflow(image_path)
    # results = object_detection_rcnn(image_path, threshold=0.7)
    # print(results)

    # boxes, classes = results
    # # print(boxes)
    
    # # # draw_bounding_boxes(image_path, boxes)
    # for box, obj_class in zip(boxes, classes):
    #     draw_bounding_box(image_path, box, obj_class)

    # # draw_bounding_box(image_path, boxes)

    # # boxes = results.boxes.xyxy[0].cpu().numpy()
    # # # boxes = np.array(boxes)
    # # print(boxes)
    # # draw_bounding_box(image_path, boxes)

    # # end_time = time.time()


