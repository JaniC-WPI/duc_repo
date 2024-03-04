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
import scipy
import matplotlib.pyplot as plt
from pos_regression import PosRegModel
from descartes import PolygonPatch

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
    print("Shape of a single configuration:", configurations[0].shape)  
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

# def sum_pairwise_euclidean(config1, config2):
#     total_distance = 0
#     for point1, point2 in zip(config1, config2):
#         total_distance += np.linalg.norm(point1 - point2)
#     return total_distance

def vectorized_sum_pairwise_euclidean(config1, config2):
    diffs = config1 - config2  # Calculate differences directly
    # print("Shape of diffs:", diffs.shape) # Add this line
    distances_squared = np.sum(diffs * diffs, axis=1)  # Adjust axis for summation 
    distances = np.sqrt(distances_squared)
    return np.sum(distances)  


def hausdorff_distance(config1, config2):
    # You might need a library like SciPy for an efficient implementation
    return scipy.spatial.distance.directed_hausdorff(config1, config2)[0] 

def heuristic(config1, config2):
    distance = vectorized_sum_pairwise_euclidean(config1, config2)
    return distance * 0.8  # Slightly underestimate the distance

def build_lazy_roadmap_with_kdtree(configurations, k_neighbors):
    """
    Build a LazyPRM roadmap using a KDTree for efficient nearest neighbor search.
    
    Args:
    - configurations: List[np.array], a list of configurations (keypoints flattened).
    - k_neighbors: int, the number of neighbors to connect to each node.
    
    Returns:
    - G: nx.Graph, the constructed roadmap.
    """
    configurations = configurations[1:9000:10]
    print("Shape of configurations before building the roadmap:", len(configurations), configurations[0].shape)

    flattened_configs = np.vstack([config.flatten() for config in configurations])
    tree = KDTree(flattened_configs)
    G = nx.Graph()
   # flattened_configs = flattened_configs[1:9000:10]
   # configurations = configurations[1:9000:10]

    for i, config in enumerate(configurations):
        G.add_node(i, configuration=config)

    for i, config in enumerate(flattened_configs):
        _, indices = tree.query([config], k=k_neighbors + 1)  # +1 to include self in results
        #indices = tree.query_radius(config.reshape(1,-1), r=20,count_only=False) # +1 to include self in results

        for j in indices[0]:  # Skip self
            if j!=i:
                G.add_edge(i, j)
        # for j in indices[0][1:]:  # Skip self
        #     distance = vectorized_sum_pairwise_euclidean(configurations[i], configurations[j]) # You'll need to define this distance calculation
        #     G.add_edge(i, j, weight=distance)
    print(G.nodes.items())
    #SG= G.subgraph(range(1,9000,100))
    pos_dict = {n[0]:n[1]["configuration"][5] for n in G.nodes.items()} 
    nx.draw_networkx(G,node_size=5,width=0.3, with_labels=False, pos=pos_dict)
    # plt.show()        
    return G, tree, pos_dict


def heuristic(config1, config2):
    distance = vectorized_sum_pairwise_euclidean(config1, config2)
    return distance * 0.8  # Slightly underestimate the distance

# Function to add a configuration to the roadmap with collision checking
def add_config_to_roadmap(config, G, tree, k_neighbors, obstacle_center, half_diagonal, safe_distance):
    # print("Shape of config being added:", config.shape)
    flattened_config = config.flatten().reshape(1, -1)
    _, indices = tree.query(flattened_config, k=k_neighbors)
    connections = 0    
    node_id = len(G.nodes)
    G.add_node(node_id, configuration=config)
    obstacle_boundary = geom.Polygon([
        (obstacle_center[0] - (half_diagonal + safe_distance), obstacle_center[1] - (half_diagonal + safe_distance)),
        (obstacle_center[0] + (half_diagonal + safe_distance), obstacle_center[1] - (half_diagonal + safe_distance)),
        (obstacle_center[0] + (half_diagonal + safe_distance), obstacle_center[1] + (half_diagonal + safe_distance)),
        (obstacle_center[0] - (half_diagonal + safe_distance), obstacle_center[1] + (half_diagonal + safe_distance)),
    ])
    
    for i in indices[0]:
        neighbor_config = G.nodes[i]['configuration']
        # print("Shape of neighbor_config:", neighbor_config.shape)
        # # Here we need to convert configurations back to their original shape for collision checking
        # x1, y1 = config.T  # Extract x and y coordinates from the first configuration 
        # x2, y2 = neighbor_config.T  # Extract x and y coordinates from the second configuration 

        # # Plot configurations as markers
        # plt.scatter(x1, y1,  marker='o', color='blue') 
        # plt.scatter(x2, y2,  marker='o', color='red')

        # # Plot the line segment
        # plt.plot([x1[0], x2[0]], [y1[0], y2[0]], color='red')

        # # Additional settings (optional)
        # plt.xlim(0, IMAGE_WIDTH)  # Adjust based on your image dimensions
        # plt.ylim(0, IMAGE_HEIGHT)
        # plt.gca().invert_yaxis()  # To match image coordinates
        # plt.title('Potential Connection') 
        # # plt.show()
        # print("New Config:", config)
        # print("Neighbor Config:", neighbor_config)
        # print("configs for collision check", type(check_config), check_config)
        if is_collision_free(config, neighbor_config, obstacle_center, half_diagonal, safe_distance):
            visualize_interactions(config, neighbor_config, obstacle_boundary)
            G.add_edge(node_id, i)
            # connections += 1
        else:
            visualize_interactions(config, neighbor_config, obstacle_boundary)

    # if connections == 0:  # If no connections were made, remove the node
    #     print("No connections were made")
    #     G.remove_node(node_id)
    #     return None
    
    pos_dict = {n[0]:n[1]["configuration"][5] for n in G.nodes.items()}     
    
    nx.draw_networkx(G,node_size=5,width=0.3, with_labels=False, pos=pos_dict)
    # plt.show()
    
    # # Update the tree with the new node for future queries
    # new_flattened_configs = np.vstack([tree.data, flattened_config])
    # new_tree = BallTree(new_flattened_configs, metric=lambda x, y: predict_custom_distance(x, y, model))

    if nx.is_connected(G):
        print("Roadmap is connected")
    else:
        print("Roadmap is disconnected")  
    
    return node_id, pos_dict

def validate_and_remove_invalid_edges(G, obstacle_center, half_diagonal, safe_distance):
    # Iterate over a copy of the edges list to avoid modification issues during iteration
    for (u, v) in list(G.edges):
        config_u = G.nodes[u]['configuration']
        config_v = G.nodes[v]['configuration']
        # Perform the collision check for the edge
        if not is_collision_free(config_u, config_v, obstacle_center, half_diagonal, safe_distance):
            # If the edge is not collision-free, remove it from the graph
            G.remove_edge(u, v)
            print(f"Removed invalid edge: {u} <-> {v}")

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

# def is_collision_free(configuration, obstacle_center, half_diagonal, safe_distance):
#     # Define the square boundary of the obstacle including the safe distance
#     obstacle_boundary = geom.Polygon([
#         (obstacle_center[0] - (half_diagonal + safe_distance), obstacle_center[1] - (half_diagonal + safe_distance)),
#         (obstacle_center[0] + (half_diagonal + safe_distance), obstacle_center[1] - (half_diagonal + safe_distance)),
#         (obstacle_center[0] + (half_diagonal + safe_distance), obstacle_center[1] + (half_diagonal + safe_distance)),
#         (obstacle_center[0] - (half_diagonal + safe_distance), obstacle_center[1] + (half_diagonal + safe_distance)),
#     ])

#     # Check each segment of the configuration for intersection with the obstacle boundary
#     for i in range(len(configuration) - 1):
#         segment = geom.LineString([configuration[i], configuration[i + 1]])
#         # print("segment by segment", segment)
#         if segment.intersects(obstacle_boundary):
#             print("collision detected in segment")
#             # If any segment intersects, the configuration is not collision-free
#             return False

#     # If no segments intersect, the configuration is collision-free
#     return True

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

def visualize_interactions(config1, config2, obstacle_boundary):
    fig, ax = plt.subplots()
    # Plot obstacle boundary
    obstacle_patch = PolygonPatch(obstacle_boundary, alpha=0.5, color="red", zorder=2)
    ax.add_patch(obstacle_patch)
    
    # Set plot limits and aspect
    ax.set_xlim([0, IMAGE_WIDTH])
    ax.set_ylim([IMAGE_HEIGHT, 0])
    ax.set_aspect('equal')

    # Visualize interactions within each configuration
    for config in [config1, config2]:
        for i in range(len(config) - 1):
            x_values, y_values = zip(*config[i:i+2])
            ax.plot(x_values, y_values, "blue", linewidth=2, solid_capstyle='round', zorder=1)

    # Visualize interactions between corresponding keypoints across configurations
    for i in range(len(config1)):
        x_values = [config1[i][0], config2[i][0]]
        y_values = [config1[i][1], config2[i][1]]
        ax.plot(x_values, y_values, "green", linewidth=2, linestyle="--", zorder=1)

    plt.show()

def visualize_interactions_path(configurations, obstacle_boundary):
    fig, ax = plt.subplots()
    # Plot obstacle boundary
    ax.add_patch(PolygonPatch(obstacle_boundary, alpha=0.5, color="red", zorder=2))
    
    # Set plot limits and aspect
    ax.set_xlim([0, IMAGE_WIDTH])
    ax.set_ylim([IMAGE_HEIGHT, 0])
    ax.set_aspect('equal')

    # Visualize paths within configurations
    for config in configurations:
        for i in range(len(config) - 1):
            segment = [config[i], config[i + 1]]
            line = geom.LineString(segment)

            # Draw the line segment
            x, y = line.xy
            ax.plot(x, y, "blue", linewidth=2, solid_capstyle='round', zorder=1)

            # Highlight if the segment intersects the obstacle
            if line.intersects(obstacle_boundary):
                ax.plot(x, y, "red", linewidth=3, solid_capstyle='round', zorder=1)

    # Visualize connections between corresponding keypoints of consecutive configurations
    for i in range(len(configurations) - 1):
        for k in range(len(configurations[i])):
            start_point = configurations[i][k]
            end_point = configurations[i + 1][k]
            line = geom.LineString([start_point, end_point])

            # Draw the connection line
            x, y = line.xy
            ax.plot(x, y, "green", linewidth=1, linestyle='--', zorder=1)

            # Highlight if the connection intersects the obstacle
            if line.intersects(obstacle_boundary):
                ax.plot(x, y, "orange", linewidth=2, linestyle='--', zorder=1)

    plt.show()

# def visualize_interactions(configurations, obstacle_boundary):
#     fig, ax = plt.subplots()
#     # Plot obstacle boundary
#     ax.add_patch(PolygonPatch(obstacle_boundary, alpha=0.5, color="red", zorder=2))
    
#     # Set plot limits and aspect
#     ax.set_xlim([0, IMAGE_WIDTH])
#     ax.set_ylim([IMAGE_HEIGHT, 0])
#     ax.set_aspect('equal')

#     # for config in configurations:
#     for i in range(len(configurations) - 1):
#         segment = [configurations[i], configurations[i + 1]]
#         line = geom.LineString(segment)
        
#         # Draw the line segment
#         x, y = line.xy
#         ax.plot(x, y, "blue", linewidth=2, solid_capstyle='round', zorder=1)
        
#         # Check for intersection and highlight if necessary
#         if line.intersects(obstacle_boundary):
#             ax.plot(x, y, "red", linewidth=3, solid_capstyle='round', zorder=1)
#             # print(f"Collision detected between points {segment[0]} and {segment[1]}")
            
#     # plt.show()

# def visualize_interactions_path(configurations, obstacle_boundary):
#     fig, ax = plt.subplots()
#     # Plot obstacle boundary
#     ax.add_patch(PolygonPatch(obstacle_boundary, alpha=0.5, color="red", zorder=2))
    
#     # Set plot limits and aspect
#     ax.set_xlim([0, IMAGE_WIDTH])
#     ax.set_ylim([IMAGE_HEIGHT, 0])
#     ax.set_aspect('equal')

#     for config in configurations:
#         for i in range(len(config) - 1):
#             segment = [config[i], config[i + 1]]
#             line = geom.LineString(segment)

#             # Draw the line segment
#             x, y = line.xy
#             ax.plot(x, y, "blue", linewidth=2, solid_capstyle='round', zorder=1)

#             # Check for intersection and highlight if necessary
#             if line.intersects(obstacle_boundary):
#                 ax.plot(x, y, "red", linewidth=3, solid_capstyle='round', zorder=1)
#                 print(f"Collision detected between points {segment[0]} and {segment[1]}")

#     plt.show()

def find_path(G, start_node, goal_node):
    """Find a path from start to goal in the roadmap G."""
    path_indices = nx.astar_path(G, source=start_node, target=goal_node)
    
    path_configurations = [G.nodes[i]['configuration'] for i in path_indices]
    return path_configurations

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
    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/path_planning_panda_regression/'  # Replace with the path to your JSON files
    num_samples = 500
    configurations = load_keypoints_from_json(directory)
    # configurations = load_and_sample_configurations(directory, num_samples)
    # Parameters for PRM
    num_neighbors = 50 # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    roadmap, tree, pos_dict = build_lazy_roadmap_with_kdtree(configurations, num_neighbors)   
    end_time = time.time()

    print("Connection of last keypoint in lazy roadmap", pos_dict)
    print("time taken to find the graph", end_time - start_time)  

    # Define start and goal configurations as numpy arrays
    start_config = np.array([[272, 437], [266, 314], [175, 261], [187, 236], [230, 108], [215, 85]]) 
    goal_config = np.array([[271, 436], [267, 313], [223, 213], [248, 199], [383, 169], [404, 147]]) 
	
    SAFE_ZONE = 50  # Safe distance from the obstacle
    obstacle_center = (380, 73)
    half_diagonal = 20
    # safe_distance = SAFE_ZONE

    obstacle_boundary = geom.Polygon([
        (obstacle_center[0] - (half_diagonal + SAFE_ZONE), obstacle_center[1] - (half_diagonal + SAFE_ZONE)),
        (obstacle_center[0] + (half_diagonal + SAFE_ZONE), obstacle_center[1] - (half_diagonal + SAFE_ZONE)),
        (obstacle_center[0] + (half_diagonal + SAFE_ZONE), obstacle_center[1] + (half_diagonal + SAFE_ZONE)),
        (obstacle_center[0] - (half_diagonal + SAFE_ZONE), obstacle_center[1] + (half_diagonal + SAFE_ZONE)),
    ])

    # Add start and goal configurations to the roadmap
    start_node, pos_dict = add_config_to_roadmap(start_config, roadmap, tree, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE)
    print("Connection for last keypoint after adding start Node", pos_dict,len(pos_dict))
    goal_node, pos_dict = add_config_to_roadmap(goal_config, roadmap, tree, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE)
    print("Connection for last keypoint after adding goal Node", pos_dict, len(pos_dict))
        
    validate_and_remove_invalid_edges(roadmap, obstacle_center, half_diagonal, SAFE_ZONE)

    # Find and print the path from start to goal
    path = find_path(roadmap, start_node, goal_node)

    output_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/physical_path_planning/scenarios/scenarios_default/phys_path_scene_06_v2'

    image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/physical_path_planning/scenarios/obstacle_image_06.png'

    if path:
        print("Path found:", path)
        visualize_interactions_path(path, obstacle_boundary)
        plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)
    else:
        print("No path found")

    
