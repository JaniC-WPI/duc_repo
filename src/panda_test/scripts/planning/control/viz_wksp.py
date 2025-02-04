#!/usr/bin/env python3

import os
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

# Define the folder where the JSON files are stored
json_folder = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged_og/'
image_folder = '/path/to/your/images'  # If images are available

# Initialize lists to store all keypoints for plotting
all_keypoints_x = []
all_keypoints_y = []

# Counter to keep track of frames
frame_count = 0

selected_indices = [3, 4, 6, 7, 8]

# Loop through all JSON files in the folder
for filename in sorted(os.listdir(json_folder)):
    if filename.endswith('.json') and not filename.endswith('.jpg') and not filename.endswith('_joint_angles.json') and not filename.endswith('_vel.json'):
        # frame_count += 1
        
        # # Only process every 10th frame
        # if frame_count % 10 != 0:
        #     continue

        file_path = os.path.join(json_folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract the keypoints from the JSON file
        keypoints = data["keypoints"]

        selected_keypoints = [keypoints[i] for i in selected_indices]

        # Separate x and y values for the selected keypoints
        keypoints_x = [kp[0][0] for kp in selected_keypoints]
        keypoints_y = [kp[0][1] for kp in selected_keypoints]
        
        all_keypoints_x.extend(keypoints_x)
        all_keypoints_y.extend(keypoints_y)
        
        # Optionally, plot the individual keypoints for each JSON on an image
        image_path = os.path.join(image_folder, data["image_rgb"])
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            for x, y in zip(keypoints_x, keypoints_y):
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.imshow("Keypoints", image)
            cv2.waitKey(100)  # Show each image for a short time

# Plot workspace coverage using all keypoints
plt.figure(figsize=(15, 15))
plt.scatter(all_keypoints_x, all_keypoints_y, s=1, c='#517119', alpha=0.9)
# plt.title('Workspace Coverage in Image Space')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
plt.gca().invert_yaxis()  # Invert y-axis for image coordinates
plt.show()

def hex_to_bgr(hex_color):
    """
    Converts a hex color code to BGR format for OpenCV.
    Args:
        hex_color (str): Hex color code (e.g., "#517119").
    Returns:
        tuple: BGR color as a tuple (e.g., (25, 113, 81)).
    """
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return rgb[::-1]  # Reverse to convert RGB to BGR

# Parameters for the plot
image_width = 640
image_height = 480
dot_radius = 1
dot_thickness = -1 
hex_color = "gray"  # Greenish color in hex
# dot_color = hex_to_bgr(hex_color)  # Convert hex to BGR
dot_color = (128, 128, 128) 

# Initialize a blank image to accumulate keypoints
workspace_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
# workspace_image.fill(255)
# workspace_image = cv2.imread('/media/jc-merlab/Crucial X9/paper_data/figure_1_images/v2/fig_for_wksp.png')

# Indices of the keypoints to be selected
selected_indices = [3, 4, 6, 7, 8]

# Loop through all JSON files in the folder
for filename in sorted(os.listdir(json_folder)):
    if filename.endswith('.json') and not filename.endswith('.jpg') and not filename.endswith('_joint_angles.json') and not filename.endswith('_vel.json'):
        file_path = os.path.join(json_folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract the keypoints from the JSON file
        keypoints = data["keypoints"]

        # Select only the desired keypoints
        selected_keypoints = [keypoints[i] for i in selected_indices]

        # Separate x and y values for the selected keypoints
        keypoints_x = [kp[0][0] for kp in selected_keypoints]
        keypoints_y = [kp[0][1] for kp in selected_keypoints]

        # Plot the keypoints on the workspace image
        for x, y in zip(keypoints_x, keypoints_y):
            cv2.circle(workspace_image, (int(x), int(y)), dot_radius, dot_color, dot_thickness)

# Optionally, save the workspace image
output_path = "/media/jc-merlab/Crucial X9/paper_data/workspace_coverage_v2.png"
cv2.imwrite(output_path, workspace_image)


# Define paths to JSON data folders
json_folder = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged_og/'
image_folder = '/path/to/your/images'  # Optional, only if images are available

# Image dimensions
image_width, image_height = 640, 480

# Initialize lists to store image space end-effector coordinates
img_end_effector_x = []
img_end_effector_y = []

# Process each JSON file for end-effector coordinates in image space
for filename in sorted(os.listdir(json_folder)):
    if filename.endswith('.json') and not filename.endswith(('.jpg', '_joint_angles.json', '_vel.json')):
        file_path = os.path.join(json_folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract only the end-effector keypoint (assuming it's the last keypoint)
        end_effector = data["keypoints"][-1][0]  # Last keypoint
        img_end_effector_x.append(end_effector[0])
        img_end_effector_y.append(end_effector[1])

# Find the min and max values of image coordinates
# image_x_min, image_x_max = 0, image_width
# image_y_min, image_y_max = 0, image_height
        
plt.figure(figsize=(8, 8))

# Plot image space end-effector coordinates
plt.scatter(img_end_effector_x, img_end_effector_y, s=1, color='#517119', alpha=0.9)

plt.gca().invert_yaxis()  # Invert y-axis for image-style display
plt.show()


# Process world coordinates data and calculate end-effector positions
# DH parameters (alpha, a, d, theta) as provided
DH_params_home = [
    [0,      0.333,             0,   0],
    [0,          0,  -(math.pi/2),   0],
    [0,      0.316,   (math.pi/2),   0],
    [0.0825,     0,   (math.pi/2),   0],
    [-0.0825, 0.384,  -(math.pi/2),  0],
    [0,          0,   (math.pi/2),   0],
    [0.088,      0,   (math.pi/2),   0],
    [0.1,      0.1070,   0,          0]]

# Initial joint positions with moving joints to be replaced
# original_joint_positions = [0, 0, 0, 0, 0, 0, 0]
original_joint_positions = [0.007195404887023141, 0, -0.008532170082858044, 0, 0.0010219530727038648, 0, 0.8118303423692146]    


def calculate_transformation_matrix(alpha, a, d, theta):
    """Calculate the transformation matrix using DH parameters."""
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0,              np.sin(alpha),                 np.cos(alpha),                d],
        [0,              0,                             0,                            1]
    ])

def forward_kinematics(joint_angles):
    """Compute the forward kinematics for the end-effector position."""
    T = np.eye(4)  # Start with identity matrix as the base transformation
    for i in range(7):
        alpha, a, d, theta_offset = DH_params_home[i]
        theta = theta_offset + joint_angles[i]
        T = np.dot(T, calculate_transformation_matrix(alpha, a, d, theta))
    return T[0, 3], T[1, 3], T[2, 3]  # Extract end-effector position

# Lists to store the world space end-effector coordinates
world_end_effector_x = []
world_end_effector_y = []
world_end_effector_z = []

# Extract and calculate world coordinates from JSON files
for filename in sorted(os.listdir(json_folder)):
    if filename.endswith('_joint_angles.json'):
        file_path = os.path.join(json_folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Replace moving joint values in the initial joint positions
        moving_joint_values = data["joint_angles"]
        full_joint_positions = original_joint_positions[:]
        full_joint_positions[1], full_joint_positions[3], full_joint_positions[5] = moving_joint_values
        
        # Compute end-effector position in world coordinates
        x, y, z = forward_kinematics(full_joint_positions)
        world_end_effector_x.append(x)
        world_end_effector_y.append(y)
        world_end_effector_z.append(z)

# Normalize world coordinates to fit within the image coordinate range
world_x_min, world_x_max = min(world_end_effector_x), max(world_end_effector_x)
world_y_min, world_y_max = min(world_end_effector_y), max(world_end_effector_y)
img_x_min, img_x_max = min(img_end_effector_x), max(img_end_effector_x)
img_y_min, img_y_max = min(img_end_effector_y), max(img_end_effector_y)

print(world_x_min, world_x_max)
print(img_x_min, img_x_max)

# Scaling factors to map world coordinates to image coordinates
scale_x = (img_x_max - img_x_min) / (world_x_max - world_x_min)
scale_y = (img_y_max - img_y_min) / (world_y_max - world_y_min)  # Negative to flip y-axis

# Centering offsets
world_x_center = (world_x_max + world_x_min) / 2
world_y_center = (world_y_max + world_y_min) / 2
# image_x_center, image_y_center = image_width / 2, image_height / 2
image_x_center, image_y_center = (img_x_max + img_x_min) / 2, (img_y_max+img_y_min) / 2


# Apply scaling and translation to world coordinates to match image space
normalized_world_x = [(x - world_x_min) * scale_x + image_x_center for x in world_end_effector_x]
normalized_world_y = [(y - world_y_min) * scale_y + image_y_center for y in world_end_effector_y]

print(min(normalized_world_x))
print(max(normalized_world_x))
print(min(normalized_world_y))
print(max(normalized_world_y))
print(img_x_min)
print(img_x_max)
print(img_y_min)
print(img_y_max)

# Plot the image and world coordinates in the same space
plt.figure(figsize=(15, 15))
plt.scatter(world_end_effector_x, world_end_effector_y, s=10, color='#5D3A9B', alpha=0.9, label='')

plt.legend('')
# plt.gca().invert_yaxis()  # Invert y-axis for image-style display
plt.axis('equal')
plt.show()


# Plot image space end-effector coordinates
# plt.scatter(img_end_effector_x, img_end_effector_y, s=1, color='green', alpha=0.5, label='Image Space')
plt.figure(figsize=(15, 15))

# Plot transformed world space end-effector coordinates
plt.scatter(normalized_world_x, normalized_world_y, s=10, color='#5D3A9B', alpha=0.9, label='')

# plt.title('Comparison of End-Effector Workspace Coverage')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
plt.legend('')
# plt.gca().invert_yaxis()  # Invert y-axis for image-style display
plt.axis('equal')
plt.show()

import pickle
import networkx as nx
import matplotlib.pyplot as plt

# Paths to the graph and KDTree files
custom_graph_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean_roadmap_angle_fresh_all_432_edges_dist_check.pkl'

# Load the graph from the .pkl file
def load_graph(graph_path):
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    print(f"Graph loaded from {graph_path}")
    return graph

# Load the graph
graph = load_graph(custom_graph_path)

# Visualize the graph with edge weights if available
def visualize_graph_with_edges(graph):
    plt.figure(figsize=(10, 8))
    
    # Get positions for nodes using spring layout
    pos = nx.spring_layout(graph)
    
    # Draw nodes and edges
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color='skyblue', edge_color='gray', font_size=10)
    
    # Draw edge labels if weights are available
    # edge_labels = nx.get_edge_attributes(graph, 'weight')  # Adjust 'weight' to your edge attribute
    # nx.draw_networkx_edge_labels(graph, pos, font_size=8)
    
    plt.title("Graph Visualization with Edges")
    # plt.show()

visualize_graph_with_edges(graph)

from pyvis.network import Network
import pickle

# Load the graph
with open(custom_graph_path, 'rb') as f:
    graph = pickle.load(f)

# Convert node identifiers to strings
graph = nx.relabel_nodes(graph, lambda x: str(x))


# Convert non-serializable attributes
for node, attributes in graph.nodes(data=True):
    for key, value in attributes.items():
        if isinstance(value, np.ndarray):
            graph.nodes[node][key] = value.tolist()

for u, v, attributes in graph.edges(data=True):
    for key, value in attributes.items():
        if isinstance(value, np.ndarray):
            graph.edges[u, v][key] = value.tolist()

# Create a Pyvis network
net = Network(notebook=False)  # Disable notebook rendering if needed
net.from_nx(graph)

# Save and view the graph in a browser
net.show("graph.html")