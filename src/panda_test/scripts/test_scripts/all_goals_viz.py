import cv2
import numpy as np
import os

# Load your image
image_path = '/home/jc-merlab/.ros/sim_published_goal_image_orig.jpg'  # Replace with the path to your image
static_image = cv2.imread(image_path)
gif_image = static_image.copy()

# Save the result
output_path = '/home/jc-merlab/.ros/sim_published_goal_image_all.jpg'
output_dir = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/14/path/'
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
        
points_sets = [[[172.0, 234.0], [192.0, 212.0], [249.0, 82.0], [248.0, 52.0], [286.0, 48.0]], [[167.0, 232.0], [188.0, 209.0], [246.0, 77.0], [264.0, 50.0], [297.0, 73.0]], [[153.0, 249.0], [169.0, 223.0], [247.0, 98.0], [268.0, 73.0], [300.0, 99.0]], [[146.0, 263.0], [159.0, 235.0], [271.0, 140.0], [271.0, 108.0], [311.0, 108.0]], [[146.0, 263.0], [159.0, 235.0], [280.0, 152.0], [304.0, 130.0], [331.0, 161.0]], [[146.0, 263.0], [159.0, 235.0], [299.0, 194.0], [326.0, 174.0], [351.0, 208.0]], [[166.0, 232.0], [187.0, 209.0], [329.0, 171.0], [357.0, 188.0], [335.0, 223.0]], [[167.0, 231.0], [188.0, 209.0], [334.0, 215.0], [367.0, 209.0], [375.0, 250.0]], [[195.0, 208.0], [222.0, 193.0], [360.0, 243.0], [393.0, 236.0], [402.0, 277.0]], [[230.0, 195.0], [260.0, 189.0], [376.0, 280.0], [410.0, 273.0], [419.0, 315.0]], [[267.0, 193.0], [298.0, 196.0], [416.0, 287.0], [448.0, 301.0], [431.0, 341.0]], [[282.0, 196.0], [312.0, 203.0], [440.0, 284.0], [474.0, 295.0], [459.0, 337.0]], [[303.0, 203.0], [331.0, 215.0], [455.0, 303.0], [488.0, 295.0], [499.0, 337.0]], [[317.0, 210.0], [343.0, 226.0], [484.0, 289.0], [519.0, 284.0], [525.0, 328.0]]]
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