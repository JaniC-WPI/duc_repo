
import json
import numpy as np
import cv2
import heapq
import os

# Parameters
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480
SAFE_DISTANCE = 30  # Safe distance from the obstacle

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

def visualize_configurations_cv(image_path, configurations, start_config, goal_config, obstacle_center):
    # Load the image
    image = cv2.imread(image_path)

    # Plot each configuration
    for config in configurations:
        for point in config:
            cv2.circle(image, tuple(point), 3, (0, 255, 255), -1)  # Yellow circles for regular configurations

    # Highlight start and goal configurations
    for point in start_config:
        cv2.circle(image, tuple(point), 5, (0, 255, 0), -1)  # Start in green
    for point in goal_config:
        cv2.circle(image, tuple(point), 5, (0, 0, 255), -1)  # Goal in red

    # Indicate the obstacle
    if obstacle_center:
        cv2.circle(image, tuple(obstacle_center), 10, (255, 0, 0), -1)  # Obstacle in blue

    # Display the image
    cv2.imshow('Configurations in Image Space', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally, save the image
    cv2.imwrite('/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/visualized_image.jpg', image)

image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/red_ball_image_5.jpg'  # Replace with the path to your image file
obstacle_info = detect_red_ball(image_path)
if obstacle_info is not None:
    obstacle_center, obstacle_radius = obstacle_info[:2], obstacle_info[2]
else:
    print("No red ball detected in the image.")
    obstacle_center, obstacle_radius = None, None

directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/kprcnn_sim_latest/'
start_config = np.array([[257, 366], [257, 283], [179, 297], [175, 276], [175, 177], [197, 181]])  # Replace with your actual start configuration
goal_config = np.array([[257, 366], [257, 283], [303, 217], [320, 229], [403, 283], [389, 297]]) 

configurations = load_keypoints_from_json(directory)


visualize_configurations_cv(image_path, configurations, start_config, goal_config, obstacle_center)
