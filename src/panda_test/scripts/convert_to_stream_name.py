#!/usr/bin/env python3

### Converts numerical file names to stream format.

import json
import os

int_stream = '000000'
folder = 9
data_dir = f'/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/marker/processed/'


if __name__ == '__main__':
    data_files = os.listdir(data_dir)  # all files in the data folder
    # filter for json files
    json_files = sorted([f for f in data_files if f.endswith('.json')])

    # use cv2 to plot each image with keypoints and bounding boxes
    for f in json_files:
        json_path = os.path.join(data_dir, f)

        with open(json_path, 'r') as f_json:
            # Figure out new name based on stream
            name, _ = os.path.splitext(f)
            new_stream = int_stream[0:-len(name)]
            new_name = new_stream + name

            data = json.load(f_json)
            # Rename image
            new_img_filename = new_name + '.rgb.jpg'
            os.rename(os.path.join(data_dir, data['image_rgb']),
                      os.path.join(data_dir, new_img_filename))
            print(f'Renaming {data["image_rgb"]} to {new_img_filename}')

        # Rename in json
        with open(json_path, 'w') as f_json:
            data['image_rgb'] = new_img_filename
            json_obj = json.dumps(data, indent=4)
            f_json.write(json_obj)

        # Rename json file
        os.rename(json_path, os.path.join(data_dir, new_name + '.json'))
