import cv2
import numpy as np

# Load your image
image_path = '/home/jc-merlab/.ros/dl_published_goal_image_obs.jpg'  # Replace with the path to your image
image = cv2.imread(image_path)

# List of points to draw
points_sets = [[[187.0, 241.0], [204.0, 219.0], [294.0, 114.0], [324.0, 116.0]], [[187.0, 241.0], [204.0, 219.0], [316.0, 141.0], [346.0, 152.0]], [[193.0, 234.0], [212.0, 213.0], [344.0, 232.0], [364.0, 257.0]]]

# [[[187.0, 241.0], [204.0, 219.0], [294.0, 114.0], [324.0, 116.0]], [[187.0, 241.0], [204.0, 219.0], [316.0, 141.0], [346.0, 152.0]]]
 # [[193.0, 234.0], [212.0, 213.0], [344.0, 232.0], [364.0, 257.0]]]
# colors = [
#     (255, 0, 0),  # Blue
#     (0, 255, 255)  # Green
#     # (255, 0, 255),  # Red
#     # (255, 255, 0)
#     # Add more colors if you have more sets of points
# ]

# [[[187.0, 241.0], [204.0, 219.0], [294.0, 114.0], [324.0, 116.0]], [[187.0, 241.0], [204.0, 219.0], [316.0, 141.0], [346.0, 152.0]], [[184.0, 246.0], [199.0, 223.0], [329.0, 191.0], [356.0, 207.0]], [[193.0, 234.0], [212.0, 213.0], [344.0, 232.0], [364.0, 257.0]], [[231.0, 207.0], [257.0, 196.0], [386.0, 245.0], [401.0, 275.0]]]

colors = np.random.randint(0, 255, (len(points_sets), 3))

# # Draw the points
# for points in points_sets:
#     for x, y in points:
#         cv2.circle(image, (int(x), int(y)), 5, (255, 0, 255), -1)  # Green points

# Draw the points with corresponding colors
# for points, color in zip(points_sets, colors):
#     for x, y in points:
#         cv2.circle(image, (int(x), int(y)), 5, color, -1)

# Draw the points with unique colors
for points, color in zip(points_sets, colors):
    for x, y in points:
        cv2.circle(image, (int(x), int(y)), 5, tuple(int(c) for c in color), -1)

# Save the result
output_path = '/home/jc-merlab/.ros/dl_published_goal_image.jpg'
cv2.imwrite(output_path, image)