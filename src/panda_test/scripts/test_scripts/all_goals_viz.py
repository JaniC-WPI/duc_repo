import cv2
import numpy as np
import os

# Load your image
image_path = '/home/jc-merlab/.ros/sim_published_goal_image_orig.jpg'  # Replace with the path to your image
static_image = cv2.imread(image_path)
gif_image = static_image.copy()

# Save the result
output_path = '/home/jc-merlab/.ros/sim_published_goal_image.jpg'
output_dir = '/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps/euc_4/path/'
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# List of points to draw
points_sets =  [[[195.0,232.0],[230.0,139.0],[243.0,120.0]],[[255.0,200.0], [330.0,140.0], [353.0,130.0]],[[255.0,200.0],[353.0,185.0],[375.0,181.0]],[[255.0,200.0],[351.0,225.0],[372.0,230.0]]]

# colors = [
#     (255, 0, 0),  # Blue
#     (0, 255, 255)  # Green
#     # (255, 0, 255),  # Red
#     # (255, 255, 0)
#     # Add more colors if you have more sets of points
# ]

# Define fixed colors for the points
fixed_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255)]

colors = np.random.randint(0, 255, (len(points_sets), 3))

for set_index, (points, color) in enumerate(zip(points_sets, colors)):
    if set_index ==0:
        for x,y in points:
            cv2.circle(static_image, (int(x), int(y)), 9, (0,255,0), -1)
        for i in range(len(points) - 1):  # Go up to the second-to-last point
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            cv2.line(static_image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), thickness=4)
    else:
        # Draw circles at each point
        for index, (x, y) in enumerate(points):
            cv2.circle(static_image, (int(x), int(y)), 9, fixed_colors[index], -1)
        for i in range(len(points) - 1):  # Go up to the second-to-last point
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            cv2.line(static_image, (int(x1), int(y1)), (int(x2), int(y2)), tuple(int(c) for c in color), thickness=4)

    cv2.imwrite(os.path.join(output_dir, f'path_{set_index}.jpg'), static_image)

cv2.imwrite(output_path, static_image)