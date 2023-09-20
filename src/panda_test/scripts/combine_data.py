#!/usr/bin/env python3

### Combines image files from source dir(s) to a destination dir.
### Automatically renames files in numerical increasing order.
### This script assumes filenames are in the format:
# n.[extension]
# where n is a number from 0

import cv2
import json
import os

if __name__ == '__main__':
    # src_dirs = [
    #     '/home/duk3/Workspace/WPI/Summer2023/ws/lama/raw_robot/dataset/train/train1/raw',
    #     '/home/duk3/Workspace/WPI/Summer2023/ws/lama/raw_robot/dataset/train/train2/raw',
    #     '/home/duk3/Workspace/WPI/Summer2023/ws/lama/raw_robot/dataset/train/train3/raw',
    #     '/home/duk3/Workspace/WPI/Summer2023/ws/lama/raw_robot/dataset/train/train4/raw',
    #     '/home/duk3/Workspace/WPI/Summer2023/ws/lama/raw_robot/dataset/train/train5/raw',
    #     '/home/duk3/Workspace/WPI/Summer2023/ws/lama/raw_robot/dataset/train/train6/raw',
    #     '/home/duk3/Workspace/WPI/Summer2023/ws/lama/raw_robot/dataset/train/train7/raw',
    #     '/home/duk3/Workspace/WPI/Summer2023/ws/lama/raw_robot/dataset/train/train8/raw',
    #     '/home/duk3/Workspace/WPI/Summer2023/ws/lama/raw_robot/dataset/train/train9/raw',
    #     '/home/duk3/Workspace/WPI/Summer2023/ws/lama/raw_robot/dataset/train/train10/raw',
    # ]

    # Combine all folders in root_src_dir:
    root_src_dir = '/home/jc-merlab/lama/predict_data/2023-09-13'
    src_dirs = [os.path.join(root_src_dir, d) for d in os.listdir(root_src_dir)]
    src_dirs = [d for d in src_dirs if os.path.isdir(d)]
    dest_dir = '/home/jc-merlab/lama/predict_data/tmp2'

    for src_dir in src_dirs:
        print(f'Processing src_dir: {src_dir}')
        src_data_files = os.listdir(src_dir)  # all files in the data folder
        # filter for image files
        src_img_files = sorted([f for f in src_data_files
                                 if f.endswith('.png') or f.endswith('.jpg')])

        dest_data_files = os.listdir(dest_dir)  # all files in the data folder
        # filter for image files
        dest_img_files = sorted([f for f in dest_data_files
                                  if f.endswith('.png') or f.endswith('.jpg')])

        for j in range(len(src_img_files)):
            # Copy image
            tmp = src_img_files[j].split('.')
            old_name, extension = tmp, '.'+'.'.join(tmp[1:])
            img_path = os.path.join(src_dir, src_img_files[j])

            new_name = j + len(dest_img_files)

            # Change name
            image = cv2.imread(img_path)
            image_file = str(new_name) + extension
            cv2.imwrite(os.path.join(dest_dir, image_file), image)
            print(f'Saved {image_file}')
