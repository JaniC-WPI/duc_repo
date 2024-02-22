#!/usr/bin/env python3

import json
import numpy as np
import cv2
import heapq
import os

# # Detect a green ball in an image
# def detect_green_ball(image_path):
#     image = cv2.imread(image_path)
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     green_lower = np.array([40, 40, 40])
#     green_upper = np.array([70, 255, 255])
#     mask = cv2.inRange(hsv, green_lower, green_upper)
#     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
#         return (int(x), int(y), int(radius))
#     return None


# if __name__ == "__main__":
#     # Detect the obstacle (green ball)
#     image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/green_ball_image.jpg'  # Replace with the path to your image file
#     image = cv2.imread(image_path)
#     obstacle_info = detect_green_ball(image_path)
#     if obstacle_info is not None:
#         obstacle_center, obstacle_radius = obstacle_info[:2], obstacle_info[2]
#         # Draw the circle on the image array
#         cv2.circle(image, tuple(obstacle_center), radius=3, color=(0,0,255), thickness=2)
#         # Save the modified image
#         cv2.imwrite("/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/obstacle_center.jpg", image)
#         # obstacle_center, obstacle_radius = obstacle_info[:2], obstacle_info[2]
#     else:
#         print("No red ball detected in the image.")
#         obstacle_center, obstacle_radius = None, None

if __name__ == "__main__":
    image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/physical_path_planning/scenarios/obstacle_image_05.png'
    start_config = np.array([[272, 437], [266, 314], [175, 261], [187, 236], [230, 108], [215, 85]]) 
    # goal_config = np.array([[257, 366], [257, 283], [183, 254], [191, 234], [287, 212], [309, 216]])
    goal_config = np.array([[271, 436], [267, 313], [223, 213], [248, 199], [383, 169], [404, 147]]) 

    base_image = cv2.imread(image_path)

    # Draw start and goal keypoints
    for point in start_config:
        cv2.circle(base_image, tuple(point.astype(int)), radius=5, color=(0, 0, 255), thickness=-1)  # Red for start
    for point in goal_config:
        cv2.circle(base_image, tuple(point.astype(int)), radius=5, color=(0, 255, 0), thickness=-1)  # Green for goal

    for i in range(len(start_config) - 1):
        cv2.line(base_image, tuple(start_config[i].astype(int)), tuple(start_config[i+1].astype(int)), (0, 0, 255), 2)
    for i in range(len(goal_config) - 1):
        cv2.line(base_image, tuple(goal_config[i].astype(int)), tuple(goal_config[i+1].astype(int)), (0, 255, 0), 2)

    cv2.imwrite('/home/jc-merlab/Pictures/panda_data/panda_sim_vel/physical_path_planning/scenarios/path_planning_phys_scene_05.jpg', base_image)

