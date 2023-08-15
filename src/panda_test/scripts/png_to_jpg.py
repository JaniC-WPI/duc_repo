#!/usr/bin/env python3

### Convert image types and update json files at the same time
### This script works with dream-like data.

from PIL import Image
import json
import os

int_stream = '000000'
folder = 9
# data_dir = f'/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/kp_test_images/{folder}/'
# data_dir = f'/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/sim_marker/raw_mask/'
data_dir = '/home/duk3/Workspace/WPI/Summer2023/ws/lama/predict_data/08_11_2023/raw_and_mask/'


if __name__ == '__main__':
    data_files = os.listdir(data_dir)  # all files in the data folder
    # filter for json files
    json_files = sorted([f for f in data_files if f.endswith('.json')])

    # use cv2 to plot each image with keypoints and bounding boxes
    for f in json_files:
        # process file names
        json_path = os.path.join(data_dir, f)

        with open(json_path, 'r') as f_json:
            data = json.load(f_json)
            name, ext = os.path.splitext(data['image_rgb'])
            new_img_filename = name + '.png'
            print(f'Renaming {data["image_rgb"]} to {new_img_filename}')
            image = Image.open(os.path.join(data_dir, data['image_rgb']))
            image.save(os.path.join(data_dir, new_img_filename))

        # Rename in json
        with open(json_path, 'w') as f_json:
            data['image_rgb'] = new_img_filename
            json_obj = json.dumps(data, indent=4)
            f_json.write(json_obj)
