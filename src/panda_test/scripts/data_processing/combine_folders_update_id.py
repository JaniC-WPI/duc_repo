import os
import glob
import json
import shutil

def update_json_id(file_path, new_id):
    with open(file_path, 'r') as file:
        data = json.load(file)
    data['id'] = new_id
    if 'image_rgb' in data:
        data['image_rgb'] = f"{new_id:06d}.jpg"
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def combine_folders(folder_paths, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    current_index = 0
    
    for folder_path in folder_paths:
        image_files = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))
        
        for image_file in image_files:
            base_name = os.path.basename(image_file).split('.')[0]
            new_base_name = f"{current_index:06d}"
            
            # Paths to old and new files
            new_image_path = os.path.join(destination_folder, f"{new_base_name}.jpg")
            old_json_path = os.path.join(folder_path, f"{base_name}.json")
            new_json_path = os.path.join(destination_folder, f"{new_base_name}.json")
            
            old_joint_angles_path = os.path.join(folder_path, f"{base_name}_joint_angles.json")
            new_joint_angles_path = os.path.join(destination_folder, f"{new_base_name}_joint_angles.json")
            
            old_vel_path = os.path.join(folder_path, f"{base_name}_vel.json")
            new_vel_path = os.path.join(destination_folder, f"{new_base_name}_vel.json")
            
            # Move and rename the image file
            shutil.copy(image_file, new_image_path)
            
            # Move and rename the JSON files if they exist, and update their ids
            if os.path.exists(old_json_path):
                shutil.copy(old_json_path, new_json_path)
                update_json_id(new_json_path, current_index)
            
            if os.path.exists(old_joint_angles_path):
                shutil.copy(old_joint_angles_path, new_joint_angles_path)
                update_json_id(new_joint_angles_path, current_index)
            
            if os.path.exists(old_vel_path):
                shutil.copy(old_vel_path, new_vel_path)
                update_json_id(new_vel_path, current_index)
            
            current_index += 1

# List of folders to combine
folders_to_combine = [
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/1_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/2_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/3_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/4_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/6_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/8_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/7_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/9_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/10_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/11_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/12_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/13_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/14_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/15_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/16_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/17_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/18_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/19_kp/',
    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/20_kp/'
]

# Destination folder where all files will be combined
destination_folder = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged_og'

combine_folders(folders_to_combine, destination_folder)
print("Folders combined successfully!")