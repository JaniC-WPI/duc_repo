import cv2
import numpy as np
import os

# Load your image
image_path = '/home/jc-merlab/.ros/sim_published_goal_image_orig.jpg'  # Replace with the path to your image
static_image = cv2.imread(image_path)
gif_image = static_image.copy()

# Save the result
output_path = '/home/jc-merlab/.ros/sim_published_goal_image_all.jpg'

# output_dir = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/58/path/'
# output_dir = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_space/42/path/'
output_dir = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/54/path/'
# output_dir = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_angle_correspondence/58/path_custom/'
# output_dir = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_angle_correspondence/58/path_euclidean/'





if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# List of points to draw
       
points_sets = [[[172.0, 234.0], [192.0, 212.0], [249.0, 82.0], [248.0, 54.0], [286.0, 48.0]], [[166.0, 232.0], [187.0, 209.0], [255.0, 81.0], [250.0, 51.0], [288.0, 44.0]], [[153.0, 249.0], [169.0, 223.0], [245.0, 97.0], [244.0, 65.0], [284.0, 64.0]], [[146.0, 263.0], [159.0, 235.0], [243.0, 115.0], [234.0, 85.0], [272.0, 75.0]], [[146.0, 263.0], [159.0, 235.0], [266.0, 135.0], [264.0, 103.0], [304.0, 101.0]], [[146.0, 263.0], [159.0, 235.0], [280.0, 152.0], [292.0, 122.0], [330.0, 137.0]], [[146.0, 263.0], [159.0, 235.0], [291.0, 172.0], [319.0, 154.0], [341.0, 189.0]], [[146.0, 263.0], [159.0, 235.0], [299.0, 194.0], [332.0, 200.0], [324.0, 241.0]], [[166.0, 232.0], [187.0, 209.0], [333.0, 190.0], [359.0, 210.0], [334.0, 243.0]], [[173.0, 225.0], [195.0, 204.0], [341.0, 220.0], [361.0, 247.0], [327.0, 272.0]], [[195.0, 208.0], [222.0, 193.0], [367.0, 220.0], [388.0, 246.0], [356.0, 273.0]], [[209.0, 202.0], [238.0, 190.0], [385.0, 203.0], [408.0, 227.0], [378.0, 257.0]], [[230.0, 195.0], [260.0, 189.0], [406.0, 213.0], [425.0, 242.0], [390.0, 265.0]], [[245.0, 193.0], [276.0, 190.0], [421.0, 226.0], [452.0, 243.0], [431.0, 281.0]], [[267.0, 193.0], [298.0, 196.0], [440.0, 245.0], [472.0, 257.0], [457.0, 297.0]], [[282.0, 196.0], [312.0, 203.0], [460.0, 238.0], [490.0, 256.0], [467.0, 294.0]], [[292.0, 198.0], [321.0, 208.0], [467.0, 253.0], [502.0, 255.0], [499.0, 299.0]], [[303.0, 203.0], [331.0, 215.0], [480.0, 250.0], [516.0, 250.0], [515.0, 294.0]], [[317.0, 210.0], [343.0, 226.0], [495.0, 252.0], [530.0, 247.0], [536.0, 291.0]]]


# Define fixed colors for the points
fixed_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

colors = np.random.randint(0, 255, (len(points_sets), 3))

for set_index, (points, color) in enumerate(zip(points_sets, colors)):
    print(points)
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