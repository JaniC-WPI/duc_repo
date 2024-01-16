#!/usr/bin/env python3

import cv2

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

# Example usage
image_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/000129.jpg'  # Replace with your image file path
ball_position = (350, 193)  # Example position (x, y)
ball_radius = 20  # Example radius
save_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/rrt_test_image/red_ball_image_5.jpg'  # Replace with your save file path

draw_red_ball(image_path, ball_position, ball_radius, save_path)