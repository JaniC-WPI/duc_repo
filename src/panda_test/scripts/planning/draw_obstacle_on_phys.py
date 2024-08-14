#!/usr/bin/env python3

import cv2
import numpy as np


import cv2
import numpy as np

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

    # Save the new image
    cv2.imwrite(save_path, image)

    print(f"Image saved with a green rectangle at center {rectangle_center} to {save_path}")

# Example usage
image_path = '/home/jc-merlab/.ros/sim_published_goal_image_all.jpg'  # Replace with your image file path
<<<<<<< HEAD
rectangle_center = (415, 140)  # Example center position (x, y)
=======
<<<<<<< HEAD
rectangle_center = (400, 120)  # Example center position (x, y)
=======
rectangle_center = (380, 120)  # Example center position (x, y)
>>>>>>> 3048ed4cf017610408a8c4c32588a17e56d0ddb6
>>>>>>> 8c1d0cff8326e15ac7b7b0d8d489e6495ce4e7fd
half_diagonal = 20  # Example half diagonal length
# save_path = '/home/jc-merlab/.ros/dl_published_goal_image.jpg'  # Replace with your save file path
save_path = '/home/jc-merlab/.ros/sim_published_goal_image.jpg'

draw_green_rectangle(image_path, rectangle_center, half_diagonal, save_path)