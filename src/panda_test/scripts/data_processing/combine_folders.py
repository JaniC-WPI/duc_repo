#!/usr/bin/env python3

import os
import json
import shutil

def update_json_file(file_path, new_id, new_image_name):
    with open(file_path, 'r') as file:
        data = json.load(file)
        data['id'] = new_id
        if 'image_rgb' in data:
            data['image_rgb'] = new_image_name
    
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def process_folder(source_folder, dest_folder, start_index):
    file_names = sorted(os.listdir(source_folder))
    jpg_files = [f for f in file_names if f.endswith('.jpg')]
    
    for jpg_file in jpg_files:
        file_number = jpg_file.split('.')[0]
        new_file_number = str(start_index).zfill(6)

        # Process .jpg file
        shutil.copy(os.path.join(source_folder, jpg_file), os.path.join(dest_folder, f"{new_file_number}.jpg"))

        # Process .json and _vel.json files
        for suffix in ['.json', '_vel.json']:
            json_file = f"{file_number}{suffix}"
            if json_file in file_names:
                update_json_file(os.path.join(source_folder, json_file), start_index, f"{new_file_number}.jpg")
                shutil.copy(os.path.join(source_folder, json_file), os.path.join(dest_folder, f"{new_file_number}{suffix}"))

        start_index += 1

    return start_index

# Define your source folders and destination folder
source_folders = ['/home/jc-merlab/Pictures/panda_data/panda_sim_vel/1/', 
                  '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/2/', 
                  '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/3/', 
                  '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/4/',
                  '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/5/',
                  '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/6/',
                  '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/7/',
                  '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/8/']
destination_folder = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/path_planning_panda/'

# Make sure the destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Process each folder
index = 0
for folder in source_folders:
    index = process_folder(folder, destination_folder, index)