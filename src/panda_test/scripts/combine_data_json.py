#!/usr/bin/env python3

### Combines data from source dir(s) to a destination dir.
### Automatically renames files in numerical increasing order.
### This script assumes files in the dream format and filenames are numbers from 0.
### json files are also updated.

import cv2
import json
import os

int_stream = '000000'
from_folder = 8
to_folder = '8_10'
# src_dir = f'/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/kp_test_images/{from_folder}/'
src_dirs = [
    '/home/jc-merlab/lama/predict_data/2023-09-13/prediction/output',
]
# dest_dir = f'/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/kp_test_images/{to_folder}/'
# dest_dir = '/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/planar_occluded/'
dest_dir = '/home/jc-merlab/lama/predict_data/2023-09-13/prediction/output-renamed'


if __name__ == '__main__':
    for src_dir in src_dirs:
        src_data_files = os.listdir(src_dir)  # all files in the data folder
        # filter for json files
        src_json_files = sorted([f for f in src_data_files if f.endswith('.json')])

        dest_data_files = os.listdir(dest_dir)  # all files in the data folder
        # filter for json files
        dest_json_files = sorted([f for f in dest_data_files if f.endswith('.json')])

        for j in range(len(src_json_files)):
            # process file names
            # Stream-named files
            # src_stream = int_stream[0:-len(str(j))]
            # json_path = src_dir + src_stream + str(j) + '.json'

            # Non-stream-named files
            json_path = os.path.join(src_dir, src_json_files[j])

            new_name = str(j + len(dest_json_files))

            # Stream-named files
            new_stream = int_stream[0:-len(new_name)]
            new_name = new_stream + new_name
            new_json_path = os.path.join(dest_dir, new_name + '.json')

            # Non-stream-named files
            # new_json_path = os.path.join(src_dir, f'{new_name}.json')

            with open(json_path, 'r') as f_json:
                data = json.load(f_json)
                # Change name
                data['id'] = int(new_name)
                # Copy image
                tmp = data['image_rgb'].split('.')
                old_name, extension = tmp, '.'+'.'.join(tmp[1:])
                image = cv2.imread(os.path.join(src_dir, data['image_rgb']))
                image_file = new_name + extension
                data['image_rgb'] = image_file
                cv2.imwrite(os.path.join(dest_dir, image_file), image)

            with open(new_json_path, 'w') as f_json:
                json_obj = json.dumps(data, indent=4)
                f_json.write(json_obj)
                print(f'Saved {new_json_path}')
