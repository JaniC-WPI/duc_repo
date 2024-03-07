import cv2
import numpy as np

# Load your image
image_path = '/home/jc-merlab/.ros/dl_published_goal_image_obs.jpg'  # Replace with the path to your image
image = cv2.imread(image_path)

# List of points to draw
points_sets = [[[167.0, 294.0], [173.0, 266.0], [160.0, 138.0], [132.0, 118.0]], [[178.0, 264.0], [191.0, 240.0], [240.0, 120.0], [224.0, 98.0]], [[178.0, 264.0], [191.0, 240.0], [297.0, 163.0], [290.0, 136.0]], [[221.0, 221.0], [245.0, 208.0], [376.0, 208.0], [388.0, 181.0]], [[280.0, 208.0], [307.0, 210.0], [408.0, 299.0], [436.0, 285.0]], [[280.0, 208.0], [307.0, 210.0], [440.0, 232.0], [459.0, 208.0]]]

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
    for i in range(len(points) - 1):  # Go up to the second-to-last point
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), tuple(int(c) for c in color), thickness=2)
    
    # Draw circles at each point
    for x, y in points:
        cv2.circle(image, (int(x), int(y)), 9, tuple(int(c) for c in color), -1)

# Save the result
output_path = '/home/jc-merlab/.ros/dl_published_goal_image.jpg'
cv2.imwrite(output_path, image)