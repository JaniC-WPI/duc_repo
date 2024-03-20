import cv2
import numpy as np
import os

# Load your image
image_path = '/home/jc-merlab/.ros/dl_published_goal_image_obs.jpg'  # Replace with the path to your image
static_image = cv2.imread(image_path)
gif_image = static_image.copy()

# Save the result
output_path = '/home/jc-merlab/.ros/dl_published_goal_image.jpg'
output_dir = '/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps/euc_4/path/'
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# List of points to draw
points_sets =  [[[272.0, 315.0], [206.0, 232.0], [228.0, 214.0], [319.0, 122.0], [331.0, 93.0]], [[268.0, 314.0], [212.0, 225.0], [235.0, 210.0], [325.0, 114.0], [354.0, 119.0]], [[268.0, 314.0], [198.0, 236.0], [218.0, 217.0], [338.0, 163.0], [355.0, 138.0]], [[268.0, 314.0], [196.0, 238.0], [215.0, 219.0], [337.0, 169.0], [358.0, 189.0]], [[268.0, 314.0], [204.0, 232.0], [225.0, 215.0], [355.0, 209.0], [382.0, 222.0]], [[268.0, 314.0], [223.0, 219.0], [247.0, 206.0], [372.0, 249.0], [393.0, 270.0]], [[268.0, 314.0], [238.0, 213.0], [264.0, 204.0], [381.0, 265.0], [412.0, 264.0]], [[268.0, 314.0], [275.0, 208.0], [302.0, 209.0], [428.0, 257.0], [434.0, 287.0]], [[268.0, 314.0], [289.0, 209.0], [316.0, 214.0], [444.0, 259.0], [451.0, 290.0]]]

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