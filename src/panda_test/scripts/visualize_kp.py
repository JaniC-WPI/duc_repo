#!/usr/bin/env python3

### Visualize robot image with keypoints.

import cv2
import json
import os
import sys
import math

int_stream = '000000'
folder = 1
# default_data_dir = f'/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/kp_test_images/{folder}/'
# default_data_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/data/kp_test_images/{folder}/'

default_data_dir = '/home/jc-merlab/Pictures/Data/Dataset_EE/Combined_dataset/'
output_data_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/kp_plan_viz/'
# def visualize(image, keypoints, wait, window_name='Default'):
#     for i in range(len(keypoints)):
#         # Draw keypoints
#         u = keypoints[i][0][0][0]
#         v = keypoints[i][0][0][1]
#         image = cv2.circle(image, (u, v), radius=3,
#                            color=(0, 0, 255), thickness=-1)
#         # # Draw bounding boxes
#         # u_bb1 = round(bboxes[i][0])
#         # v_bb1 = round(bboxes[i][1])
#         # u_bb2 = round(bboxes[i][2])
#         # v_bb2 = round(bboxes[i][3])
#         # image = cv2.rectangle(image, (u_bb1, v_bb1), (u_bb2, v_bb2),
#         #                       color=(0, 255, 0), thickness=1)

#     cv2.imshow(window_name, image)
#     cv2.waitKey(wait)


# if __name__ == '__main__':
#     data_dir = sys.argv[1] if len(sys.argv) > 1 else default_data_dir
#     data_files = os.listdir(data_dir)  # all files in the data folder
#     # filter for json files
#     json_files = sorted([f for f in data_files if f.endswith('.json')])

#     # use cv2 to plot each image with keypoints and bounding boxes
#     for json_file in json_files:
#         # process file names
#         # new_stream = int_stream[0:-len(str(j))]
#         # json_path = os.path.join(data_dir, new_stream + str(j) + '.json')
#         json_path = os.path.join(data_dir, json_file)

#         with open(json_path, 'r') as f_json:
#             data = json.load(f_json)
#             image = cv2.imread(os.path.join(data_dir, data['image_rgb']))
#             # print kp visibility
#             # kp_vis = [kp[0][2] for kp in data['keypoints']]
#             # print(kp_vis)

#             visualize(image, data['keypoints'], 0, json_file)
#             cv2.destroyWindow(json_file)
#     cv2.destroyAllWindows()
# def visualize_and_save(image, keypoints, filename, output_data_dir):
#     for sublist in keypoints:  # Iterate through sublists 
#         print(sublist)
#         for kp in sublist:  # Iterate through keypoints within each sublist
#             print(kp)
#             u = int(kp[0])    
#             print(u)
#             v = int(kp[1])    
#             image = cv2.circle(image, (u, v), radius=3, color=(0, 0, 255), thickness=-1)


#     # Save visualized image
#     output_path = os.path.join(output_data_dir, filename)
#     cv2.imwrite(output_path, image)


# if __name__ == '__main__':
#     data_dir = sys.argv[1] if len(sys.argv) > 1 else default_data_dir
#     # json_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json") and not f.endswith("_vel.json")
#     #                       and not f.endswith("_combined.json")  and not f.endswith("_joint_angles.json")])
    
#     json_files = sorted([f for f in os.listdir(default_data_dir) if f.endswith(".json")])

#     for json_file in json_files:
#         json_path = os.path.join(default_data_dir, json_file)

#         with open(json_path, 'r') as f_json:
#             data = json.load(f_json)

#         # Load the corresponding image
#         image_filename = data['image_rgb']
#         image_path = os.path.join(default_data_dir, image_filename)
#         image = cv2.imread(image_path)

#         # Visualize and save
#         visualize_and_save(image, data['keypoints'], image_filename, output_data_dir) 


import cv2
import json
import os
import sys

# Paths to the input and output directories
default_data_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/path_planning_kp_sim/'
output_data_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/kp_plan_viz/'

def visualize_and_save(image, keypoints, filename, output_data_dir):
    # Iterate through all keypoints and draw them on the image
    for sublist in keypoints:
        for kp in sublist:
            u = int(kp[0])
            v = int(kp[1])
            image = cv2.circle(image, (u, v), radius=3, color=(0, 0, 255), thickness=-1)

    # Save the image to the specified output directory
    output_path = os.path.join(output_data_dir, filename)
    cv2.imwrite(output_path, image)

if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_data_dir

    # Ensure output directory exists
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    # List JSON files
    json_files = sorted([f for f in os.listdir(data_dir) \
                         if f.endswith(".json") and not f.endswith("_combined.json") \
                            and not f.endswith("_vel.json")])

    for json_file in json_files:
        base_filename = json_file.split('.')[0]  # Remove the extension from the json filename
        image_filename = f"{base_filename}.jpg"  # Construct the corresponding image filename
        
        json_path = os.path.join(data_dir, json_file)
        image_path = os.path.join(data_dir, image_filename)

        # Check if the corresponding image file exists
        if os.path.exists(image_path):
            with open(json_path, 'r') as f_json:
                data = json.load(f_json)

            image = cv2.imread(image_path)

            # Visualize keypoints on the image and save in the output directory
            visualize_and_save(image, data['keypoints'], image_filename, output_data_dir)
        else:
            print(f"No corresponding image found for {json_file}")