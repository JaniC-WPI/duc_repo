import os
import glob

# Path to your folder containing the images and JSON files
folder_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/19_kp_cleaned/'

# Get all image files and sort them
image_files = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))

# Function to get the base name without extension
def get_base_name(file_path):
    return os.path.basename(file_path).split('.')[0]

# Get the unique base names from image files
base_names = [get_base_name(image) for image in image_files]

# Renaming the files sequentially
for idx, base_name in enumerate(base_names):
    new_base_name = f"{idx:06d}"
    
    # Paths to old and new files
    old_image_path = os.path.join(folder_path, f"{base_name}.jpg")
    new_image_path = os.path.join(folder_path, f"{new_base_name}.jpg")
    
    old_json_path = os.path.join(folder_path, f"{base_name}.json")
    new_json_path = os.path.join(folder_path, f"{new_base_name}.json")
    
    old_joint_angles_path = os.path.join(folder_path, f"{base_name}_joint_angles.json")
    new_joint_angles_path = os.path.join(folder_path, f"{new_base_name}_joint_angles.json")
    
    old_vel_path = os.path.join(folder_path, f"{base_name}_vel.json")
    new_vel_path = os.path.join(folder_path, f"{new_base_name}_vel.json")
    
    # Rename the image file
    os.rename(old_image_path, new_image_path)
    
    # Rename the JSON files if they exist
    if os.path.exists(old_json_path):
        os.rename(old_json_path, new_json_path)
    
    if os.path.exists(old_joint_angles_path):
        os.rename(old_joint_angles_path, new_joint_angles_path)
    
    if os.path.exists(old_vel_path):
        os.rename(old_vel_path, new_vel_path)

print("Files renamed successfully!")