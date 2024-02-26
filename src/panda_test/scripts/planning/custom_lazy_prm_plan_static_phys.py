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
from pos_regression import PosRegModel
import matplotlib.pyplot as plt
from descartes import PolygonPatch


# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

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
    # print("Shape of a single configuration:", configurations[0].shape)  
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
    model = PosRegModel(12)
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

def visualize_interactions(configurations, obstacle_boundary):
    fig, ax = plt.subplots()
    # Plot obstacle boundary
    ax.add_patch(PolygonPatch(obstacle_boundary, alpha=0.5, color="red", zorder=2))
    
    # Set plot limits and aspect
    ax.set_xlim([0, IMAGE_WIDTH])
    ax.set_ylim([IMAGE_HEIGHT, 0])
    ax.set_aspect('equal')

    # for config in configurations:
    for i in range(len(configurations) - 1):
        segment = [configurations[i], configurations[i + 1]]
        line = geom.LineString(segment)
        
        # Draw the line segment
        x, y = line.xy
        ax.plot(x, y, "blue", linewidth=2, solid_capstyle='round', zorder=1)
        
        # Check for intersection and highlight if necessary
        if line.intersects(obstacle_boundary):
            ax.plot(x, y, "red", linewidth=3, solid_capstyle='round', zorder=1)
            # print(f"Collision detected between points {segment[0]} and {segment[1]}")
            
    plt.show()

def visualize_interactions_path(configurations, obstacle_boundary):
    fig, ax = plt.subplots()
    # Plot obstacle boundary
    ax.add_patch(PolygonPatch(obstacle_boundary, alpha=0.5, color="red", zorder=2))
    
    # Set plot limits and aspect
    ax.set_xlim([0, IMAGE_WIDTH])
    ax.set_ylim([IMAGE_HEIGHT, 0])
    ax.set_aspect('equal')

    for config in configurations:
        for i in range(len(config) - 1):
            segment = [config[i], config[i + 1]]
            line = geom.LineString(segment)

            # Draw the line segment
            x, y = line.xy
            ax.plot(x, y, "blue", linewidth=2, solid_capstyle='round', zorder=1)

            # Check for intersection and highlight if necessary
            if line.intersects(obstacle_boundary):
                ax.plot(x, y, "red", linewidth=3, solid_capstyle='round', zorder=1)
                # print(f"Collision detected between points {segment[0]} and {segment[1]}")

    plt.show()

class CustomDistanceHeuristic:
    def __init__(self, model, graph):
        self.model = model
        self.graph = graph

    def __call__(self, current_node, target_node):
        current_config = self.graph.nodes[current_node]['configuration']
        target_config = self.graph.nodes[target_node]['configuration']
        distance = predict_custom_distance(current_config, target_config, self.model)
        return distance

def build_lazy_roadmap_with_kdtree(configurations, k_neighbors, model):
    """
    Build a LazyPRM roadmap using a KDTree for efficient nearest neighbor search.
    
    Args:
    - configurations: List[np.array], a list of configurations (keypoints flattened).
    - k_neighbors: int, the number of neighbors to connect to each node.
    
    Returns:
    - G: nx.Graph, the constructed roadmap.
    """
    configurations = configurations[1:9000:10]
    # print("Shape of configurations before building the roadmap:", len(configurations), configurations[0].shape)

    flattened_configs = np.vstack([config.flatten() for config in configurations])
    tree = BallTree(flattened_configs, metric=lambda x, y: predict_custom_distance(x, y, model))
    print("tree is built")
    # tree = KDTree(distance_matrix) 

    G = nx.Graph()
   # flattened_configs = flattened_configs[1:9000:10]
   # configurations = configurations[1:9000:10]

    for i, config in enumerate(configurations):
        G.add_node(i, configuration=config)

    for i, config in enumerate(flattened_configs):
        _, indices = tree.query([config], k=k_neighbors + 1)  # +1 to include self in results
        #indices = tree.query_radius(config.reshape(1,-1), r=20,count_only=False) # +1 to include self in results

        for j in indices[0]:  # Skip self
            if j !=i:
                G.add_edge(i, j)
        
    # print(G.nodes.items())
    #SG= G.subgraph(range(1,9000,100))
    pos_dict = {n[0]:n[1]["configuration"][5] for n in G.nodes.items()}      
    # print(pos_dict) 
    nx.draw_networkx(G,node_size=5,width=0.3, with_labels=False, pos=pos_dict)
    # plt.show()        
    return G, tree


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
        # Here we need to convert configurations back to their original shape for collision checking
        x1, y1 = config.T  # Extract x and y coordinates from the first configuration 
        x2, y2 = neighbor_config.T  # Extract x and y coordinates from the second configuration 

        # # Plot configurations as markers
        # plt.scatter(x1, y1,  marker='o', color='blue') 
        # plt.scatter(x2, y2,  marker='o', color='blue')

        # # Plot the line segment
        # plt.plot([x1[0], x2[0]], [y1[0], y2[0]], color='red')

        # # Additional settings (optional)
        # plt.xlim(0, IMAGE_WIDTH)  # Adjust based on your image dimensions
        # plt.ylim(0, IMAGE_HEIGHT)
        # plt.gca().invert_yaxis()  # To match image coordinates
        # plt.title('Potential Connection') 
        # plt.show()
        # print("New Config:", config)
        # print("Neighbor Config:", neighbor_config)
        # print("Collision Check Result:", is_collision_free(np.vstack([config, neighbor_config]), obstacle_center, half_diagonal, safe_distance))
        check_config = np.vstack([config, neighbor_config])
        # print("configs for collision check", type(check_config), check_config)
        if is_collision_free(np.vstack([config, neighbor_config]), obstacle_center, half_diagonal, safe_distance):
            # visualize_interactions(np.array(check_config), obstacle_boundary)
            G.add_edge(node_id, i)
        # else:
            # visualize_interactions(np.array(check_config), obstacle_boundary)

    # if connections == 0:  # If no connections were made, remove the node
    #     print("No connections were made")
    #     G.remove_node(node_id)
    #     return None
    
    # # Update the tree with the new node for future queries
    # new_flattened_configs = np.vstack([tree.data, flattened_config])
    # new_tree = BallTree(new_flattened_configs, metric=lambda x, y: predict_custom_distance(x, y, model))

    if nx.is_connected(G):
        print("Roadmap is connected")
    else:
        print("Roadmap is disconnected")  
    
    return node_id

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

# def is_collision_free(configuration, obstacle_center, safe_distance, half_diagonal):
#     obstacle = square_obstacle(obstacle_center, half_diagonal + safe_distance)

#     for i in range(len(configuration) - 1):
#         line_segment = geom.LineString([configuration[i], configuration[i + 1]])
#         if line_segment.distance(obstacle) <= 0:  # Collision! 
#             return False

#     return True  # Collision-free 

def is_collision_free(configuration, obstacle_center, half_diagonal, safe_distance):
    # Define the square boundary of the obstacle including the safe distance
    obstacle_boundary = geom.Polygon([
        (obstacle_center[0] - (half_diagonal + safe_distance), obstacle_center[1] - (half_diagonal + safe_distance)),
        (obstacle_center[0] + (half_diagonal + safe_distance), obstacle_center[1] - (half_diagonal + safe_distance)),
        (obstacle_center[0] + (half_diagonal + safe_distance), obstacle_center[1] + (half_diagonal + safe_distance)),
        (obstacle_center[0] - (half_diagonal + safe_distance), obstacle_center[1] + (half_diagonal + safe_distance)),
    ])

    # Check each segment of the configuration for intersection with the obstacle boundary
    for i in range(len(configuration) - 1):
        segment = geom.LineString([configuration[i], configuration[i + 1]])
        print("segment by segment", segment)
        if segment.intersects(obstacle_boundary):
            print("collision detected in segment")
            # If any segment intersects, the configuration is not collision-free
            return False

    # If no segments intersect, the configuration is collision-free
    return True

def validate_and_remove_invalid_edges(G, obstacle_center, half_diagonal, safe_distance):
    # Iterate over a copy of the edges list to avoid modification issues during iteration
    for (u, v) in list(G.edges):
        config_u = G.nodes[u]['configuration']
        config_v = G.nodes[v]['configuration']
        # Perform the collision check for the edge
        if not is_collision_free(np.vstack([config_u, config_v]), obstacle_center, half_diagonal, safe_distance):
            # If the edge is not collision-free, remove it from the graph
            G.remove_edge(u, v)
            print(f"Removed invalid edge: {u} <-> {v}")

def find_path_heuristic(G, start_node, goal_node, heuristic):
    try:
        path_indices = nx.astar_path(G, source=start_node, target=goal_node, heuristic=heuristic, weight='cost')
        path_configurations = [G.nodes[i]['configuration'] for i in path_indices]
        return path_configurations
    except nx.NetworkXNoPath:
        print("No path found between the specified nodes.")
        return []
    
def find_path(G, start_node, goal_node):
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
    model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b64_e400_v6.pth'
    num_samples = 500
    configurations = load_keypoints_from_json(directory)
    model = load_model_for_inference(model_path)
    # distance_matrix = calculate_model_distances(configurations, model)
    # distance_matrix = np.array([1.0]).reshape(-1,1)
    # configurations = load_and_sample_configurations(directory, num_samples)
    # Parameters for PRM
    num_neighbors = 50 # Number of neighbors for each node in the roadmap
    start_time = time.time()
    # Build the roadmap
    roadmap, tree = build_lazy_roadmap_with_kdtree(configurations, num_neighbors, model)   
    end_time = time.time()

    print("time taken to find the graph", end_time - start_time)      

    # Define start and goal configurations as numpy arrays
    start_config = np.array([[271, 431], [270, 313], [194, 240], [214, 221], [300, 124], [312, 95]]) 
    goal_config = np.array([[271, 431], [271, 313], [243, 211], [270, 203], [389, 258], [418, 243]]) 
    
    SAFE_ZONE = 50  # Safe distance from the obstacle
    obstacle_center = (420, 133)
    half_diagonal = 50
    # safe_distance = SAFE_ZONE

    obstacle_boundary = geom.Polygon([
        (obstacle_center[0] - (half_diagonal + SAFE_ZONE), obstacle_center[1] - (half_diagonal + SAFE_ZONE)),
        (obstacle_center[0] + (half_diagonal + SAFE_ZONE), obstacle_center[1] - (half_diagonal + SAFE_ZONE)),
        (obstacle_center[0] + (half_diagonal + SAFE_ZONE), obstacle_center[1] + (half_diagonal + SAFE_ZONE)),
        (obstacle_center[0] - (half_diagonal + SAFE_ZONE), obstacle_center[1] + (half_diagonal + SAFE_ZONE)),
    ])

    # Add start and goal configurations to the roadmap
    start_node = add_config_to_roadmap(start_config, roadmap, tree, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE)
    goal_node = add_config_to_roadmap(goal_config, roadmap, tree, num_neighbors, obstacle_center, half_diagonal, SAFE_ZONE)

    validate_and_remove_invalid_edges(roadmap, obstacle_center, half_diagonal, SAFE_ZONE)
        
    heuristic = CustomDistanceHeuristic(model, roadmap)
    # Find and print the path from start to goal
    path = find_path(roadmap, start_node, goal_node)
    # path = find_path_heuristic(roadmap, start_node, goal_node, heuristic)

    output_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/physical_path_planning/scenarios/scenarios_custom/phys_path_no_cost_scene_02/'

    image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/physical_path_planning/scenarios/obstacle_image_02.png'

    if path:
        print("Path found:", path)
        # Convert path configurations for visualization
        # configurations_for_visualization = [roadmap.nodes[node]['configuration'] for node in path]
        # Call the visualization function
        visualize_interactions_path(path, obstacle_boundary)
        plot_path_on_image_dir(image_path, path, start_config, goal_config, output_dir)
    else:
        print("No path found")

