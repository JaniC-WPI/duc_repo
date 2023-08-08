#!/usr/bin/env python3

### Visualize robot image with keypoints.

import cv2
import json
import os
import sys

int_stream = '000000'
folder = 8
default_data_dir = f'/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/kp_test_images/{folder}/'


def visualize(image, keypoints, bboxes, wait, window_name='Default'):
    for i in range(len(keypoints)):
        # Draw keypoints
        u = round(keypoints[i][0][0])
        v = round(keypoints[i][0][1])
        image = cv2.circle(image, (u, v), radius=3,
                           color=(0, 0, 255), thickness=-1)
        # Draw bounding boxes
        u_bb1 = round(bboxes[i][0])
        v_bb1 = round(bboxes[i][1])
        u_bb2 = round(bboxes[i][2])
        v_bb2 = round(bboxes[i][3])
        image = cv2.rectangle(image, (u_bb1, v_bb1), (u_bb2, v_bb2),
                              color=(0, 255, 0), thickness=1)

    cv2.imshow(window_name, image)
    cv2.waitKey(wait)


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_data_dir
    data_files = os.listdir(data_dir)  # all files in the data folder
    # filter for json files
    json_files = sorted([f for f in data_files if f.endswith('.json')])

    # use cv2 to plot each image with keypoints and bounding boxes
    for j in range(len(json_files)):
        # process file names
        new_stream = int_stream[0:-len(str(j))]
        json_path = os.path.join(data_dir, new_stream + str(j) + '.json')

        with open(json_path, 'r') as f_json:
            data = json.load(f_json)
            image = cv2.imread(os.path.join(data_dir, data['image_rgb']))

            visualize(image, data['keypoints'], data['bboxes'], 0, f'image{j}')
            cv2.destroyWindow(f'image{j}')
    cv2.destroyAllWindows()
