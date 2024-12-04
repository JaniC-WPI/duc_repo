import os
import cv2
import numpy as np
import random

# Define points for each configuration
points = [[[169, 235], [188, 212], [244, 81], [242, 50], [280, 47]],
        [[146, 263], [159, 235], [291, 172], [323, 164], [333, 204]],
        [[282, 196], [310, 203], [441, 282], [476, 290], [466, 333]],
        [[297, 201], [324, 212], [472, 219], [500, 199], [525, 231]]]

# Yellow rectangle parameters
rect_center = (402, 139)
half_diagonal = 20

diagonal = 2 * half_diagonal
width = int(diagonal / np.sqrt(2))
height = width  # Assuming the rectangle is a square for simplicity
# Calculate top left and bottom right points of the rectangle
top_left = (rect_center[0] - width // 2, rect_center[1] - height // 2)
bottom_right = (rect_center[0] + width // 2, rect_center[1] + height // 2)

# Image size
# Assuming you have the images in separate variables
image1 = cv2.imread("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_with_obs/2/raw/7529.png")  # Load your first image here
image2 = cv2.imread("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_with_obs/2/raw/3126.png")  # Load your second image here
image3 = cv2.imread("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_with_obs/2/raw/338.png")  # Load your third image here
image4 = cv2.imread("/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_with_obs/2/raw/76.png")  # Load your fourth image here

image1 = np.zeros((480, 640, 3), np.uint8)
image2 = np.zeros((480, 640, 3), np.uint8)
image3 = np.zeros((480, 640, 3), np.uint8)
image4 = np.zeros((480, 640, 3), np.uint8)

# Put the images in a list to iterate through
images = [image1, image2, image3, image4]

# colors = ['green', '#1B0FEF', '#CE1EA6', 'red'] 
colors = [(0, 255, 0), (198, 20, 255), (255, 0, 0), (0, 0, 255)]


# Directory to save images
output_dir = "/media/jc-merlab/Crucial X9/paper_data/figure_1_images/v2/fig_path/"
os.makedirs(output_dir, exist_ok=True)

# Process each image with corresponding point set
for i, (image, config) in enumerate(zip(images, points)):
    # Make a copy to avoid modifying the original image
    img_copy = image.copy()

    # Draw a solid yellow rectangle
    cv2.rectangle(img_copy, top_left, bottom_right, (0, 255, 255), -1)

    # Choose color based on index (start, intermediate, goal)
    color = colors[i]

    # Draw lines and points for the configuration
    for j in range(len(config) - 1):
        pt1 = tuple(config[j])
        pt2 = tuple(config[j + 1])

        # Draw line between points
        cv2.line(img_copy, pt1, pt2, color, 3)

        # Draw points
        cv2.circle(img_copy, pt1, 10, color, -1)

    # Draw the final point of each configuration
    cv2.circle(img_copy, tuple(config[-1]), 10, color, -1)

    # Save the image
    image_filename = f"{output_dir}/modified_image_{i+1}.png"
    cv2.imwrite(image_filename, img_copy)

print("Images saved with configurations drawn.")