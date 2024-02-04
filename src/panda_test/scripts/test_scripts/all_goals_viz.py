import cv2

# Load your image
image_path = '/home/jc-merlab/.ros/sim_published_goal_image.jpg'  # Replace with the path to your image
image = cv2.imread(image_path)

# List of points to draw
points_sets = [
    [[175.0,277.0],[228.0,194.0],[250.0,193.0]],
    [[194.0,231.0],[291.0,214.0],[293.0,234.0]],
    [[215.0,212.0],[312.0,229.0],[307.0,249.0]],
    [[279.0,203.0],[366.0,250.0],[355.0,268.0]]
]
# Draw the points
for points in points_sets:
    for x, y in points:
        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)  # Green points

# Save the result
output_path = '/home/jc-merlab/.ros/sim_published_goal_image.jpg'
cv2.imwrite(output_path, image)