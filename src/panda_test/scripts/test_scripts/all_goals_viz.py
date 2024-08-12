import cv2
import numpy as np
import os

# Load your image
image_path = '/home/jc-merlab/.ros/sim_published_goal_image_orig.jpg'  # Replace with the path to your image
static_image = cv2.imread(image_path)
gif_image = static_image.copy()

# Save the result
output_path = '/home/jc-merlab/.ros/sim_published_goal_image_all.jpg'
output_dir = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/10/path/'
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# List of points to draw
# points_sets =  [[[255, 200], [333, 140], [354, 143], [352, 164]],\
#                 # [[255, 200], [304, 198], [354, 197], [376, 195], [377, 218]]] #,\
#                 [[280, 204], [378, 219], [392, 203], [409, 217]], \
#                 [[320, 229], [413, 266], [434, 274], [426, 295]]] #, \
#                 # [[280, 203], [315, 238], [351, 272], [366, 288], [351, 303]]

# points_sets = [[[255, 200], [353, 214], [375, 217], [372, 239]],
#                 [[255, 200], [333, 140], [352, 149], [344, 168]],
#                 [[230, 205], [269, 114], [279, 94], [299, 103]]]
        
# points_sets = [[[235, 206], [255, 200], [333, 140], [354, 143], [352, 164]],
#             #    [[235, 206], [255, 200], [354, 197], [376, 195], [377, 218]],
#                [[260, 203], [280, 204], [378, 219], [392, 203], [409, 217]],
#                [[303, 217], [320, 229], [413, 266], [434, 274], [426, 295]]]
        
# points_sets = [[[278, 203], [362, 151], [379, 163], [367, 180]],
#                [[279, 203], [370, 167], [391, 159], [399, 180]]]
            #    [[279, 203], [378, 203], [392, 218], [373, 236]],
            #    [[279, 203], [364, 254], [372, 274], [347, 284]],
            #    [[319, 228], [417, 234], [440, 232], [441, 260]],
            #    [[339, 274], [437, 263], [460, 259], [465, 286]]]

    # [[255, 200], [333, 140], [354, 144], [350, 165]]
        
points_sets = [[[294.0, 200.0], [322.0, 209.0], [468.0, 181.0], [494.0, 158.0], [522.0, 187.0]],
[[267.0, 193.0], [298.0, 196.0], [441.0, 241.0], [469.0, 223.0], [492.0, 260.0]],
[[230.0, 195.0], [260.0, 189.0], [405.0, 218.0], [432.0, 198.0], [458.0, 231.0]],
[[195.0, 208.0], [222.0, 193.0], [369.0, 196.0], [394.0, 173.0], [422.0, 204.0]],
[[195.0, 208.0], [222.0, 193.0], [369.0, 196.0], [400.0, 182.0], [418.0, 220.0]],
[[166.0, 232.0], [187.0, 209.0], [333.0, 190.0], [366.0, 184.0], [372.0, 226.0]],
[[146.0, 263.0], [159.0, 235.0], [299.0, 194.0], [331.0, 183.0], [344.0, 222.0]],
[[135.0, 299.0], [139.0, 268.0], [210.0, 140.0], [225.0, 111.0], [261.0, 130.0]],
[[135.0, 299.0], [139.0, 268.0], [229.0, 152.0], [241.0, 122.0], [278.0, 136.0]],
[[146.0, 263.0], [159.0, 235.0], [280.0, 151.0], [283.0, 119.0], [323.0, 123.0]],
[[153.0, 249.0], [169.0, 223.0], [231.0, 90.0], [227.0, 59.0], [266.0, 53.0]]]
# 
# colors = [
#     (255, 0, 0),  # Blue
#     (0, 255, 255)  # Green
#     # (255, 0, 255),  # Red
#     # (255, 255, 0)
#     # Add more colors if you have more sets of points
# ]

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