#!/usr/bin/env python3

"""
Update the visibility of keypoints in the occlusion dataset.
If the keypoint pixel is black, visibility is 0, otherwise 1.
"""

import cv2
import numpy as np
import json
import os
import sys

int_stream = '000000'
folder = 8
default_data_dir = f'/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/kp_test_images/{folder}/'
default_data_dir = '/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/planar_occluded'


def update_visibility(image, keypoints):
    for i in range(len(keypoints)):
        # Draw keypoints
        u = round(keypoints[i][0][0])
        v = round(keypoints[i][0][1])
        if np.any(image[v, u] != 0):
            keypoints[i][0][2] = 1
        else:
            keypoints[i][0][2] = 0
        print(image[v,u])


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_data_dir
    data_files = os.listdir(data_dir)  # all files in the data folder
    # filter for json files
    json_files = sorted([f for f in data_files if f.endswith('.json')])

    # use cv2 to plot each image with keypoints and bounding boxes
    for json_file in json_files:
        # process file names
        # new_stream = int_stream[0:-len(str(j))]
        # json_path = os.path.join(data_dir, new_stream + str(j) + '.json')
        json_path = os.path.join(data_dir, json_file)

        with open(json_path, 'r') as f_json:
            data = json.load(f_json)
            image = cv2.imread(os.path.join(data_dir, data['image_rgb']))
            update_visibility(image, data['keypoints'])
            print(f'Updated visibility for file {json_file}')

        with open(json_path, 'w') as f_json:
            json_obj = json.dumps(data, indent=4)
            f_json.write(json_obj)
