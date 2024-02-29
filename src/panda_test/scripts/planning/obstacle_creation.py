#!/usr/bin/env python3

import cv2
import numpy as np

SAFE_DISTANCE = 50

def draw_red_ball(image_path, ball_position, ball_radius, save_path):
    # Read the image
    image = cv2.imread(image_path)

    # Define the color of the ball (Green in BGR format)
    red_color = (0, 0, 255)

    # Draw the circle on the image
    cv2.circle(image, ball_position, ball_radius, red_color, -1)

    # Save the new image
    cv2.imwrite(save_path, image)

    print(f"Image saved with a red ball at {ball_position} to {save_path}")

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

    # Define the color of the circle (Red in BGR format)
    red_color = (0, 0, 255)
    top_left = (rectangle_center[0] - half_diagonal - SAFE_DISTANCE, 
                rectangle_center[1] - half_diagonal - SAFE_DISTANCE)
    bottom_right = (rectangle_center[0] + half_diagonal + SAFE_DISTANCE, 
                    rectangle_center[1] + half_diagonal + SAFE_DISTANCE)
    # Draw the circle around the rectangle. Set thickness to 2 (or another value) instead of -1 to not fill the circle
    # cv2.rectangle(image, top_left, bottom_right, red_color, thickness=2)

    # Save the new image
    cv2.imwrite(save_path, image)

    print(f"Image saved with a green rectangle at center {rectangle_center} to {save_path}")

# Example usage
# image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/physical_path_planning/scenarios/goal_image_2.png'
image_path = '/home/jc-merlab/.ros/dl_published_goal_image_orig.jpg'  # Replace with your image file path
      # Replace with your image file path
rectangle_center = (420, 133)  # Example center position (x, y)
half_diagonal = 20  # Example half diagonal length
# save_path = '/home/jc-merlab/.ros/dl_published_goal_image.jpg'  # Replace with your save file path
save_path = '/home/jc-merlab/.ros/dl_published_goal_image_obs.jpg'

draw_green_rectangle(image_path, rectangle_center, half_diagonal, save_path)

# Example usage
# image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/000129.jpg'  # Replace with your image file path
# ball_position = (350, 193)  # Example position (x, y)
# ball_radius = 20  # Example radius
# save_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/red_ball_image_5.jpg'  # Replace with your save file path

# draw_red_ball(image_path, ball_position, ball_radius, save_path)