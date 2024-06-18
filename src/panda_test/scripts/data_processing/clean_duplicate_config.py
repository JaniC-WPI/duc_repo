import os
import json
import hashlib

def hash_joint_angles(joint_angles):
    """Create a hash from a list of joint angles."""
    return hashlib.md5(json.dumps(joint_angles, sort_keys=True).encode('utf-8')).hexdigest()

def remove_duplicates_and_rename(directory):
    seen_joint_angles = {}
    files_to_delete = []
    unique_files = []

    for filename in os.listdir(directory):
        if filename.endswith('_joint_angles.json'):
            base_name = filename.replace('_joint_angles.json', '')
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                joint_angles = data['joint_angles']
                angles_hash = hash_joint_angles(joint_angles)
                
                if angles_hash in seen_joint_angles:
                    files_to_delete.append(f"{base_name}.jpg")
                    files_to_delete.append(f"{base_name}.json")
                    files_to_delete.append(f"{base_name}_joint_angles.json")
                    files_to_delete.append(f"{base_name}_vel.json")
                else:
                    seen_joint_angles[angles_hash] = filename
                    unique_files.append(base_name)
    
    # Remove duplicates
    for filename in files_to_delete:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")

    # Rename the remaining files consecutively
    for new_id, base_name in enumerate(sorted(unique_files)):
        new_base_name = f"{new_id:06d}"
        
        for suffix in ['.jpg', '.json', '_joint_angles.json', '_vel.json']:
            old_file = os.path.join(directory, f"{base_name}{suffix}")
            new_file = os.path.join(directory, f"{new_base_name}{suffix}")
            if os.path.exists(old_file):
                os.rename(old_file, new_file)
                print(f"Renamed: {old_file} to {new_file}")
        
        # Update the JSON files with the new id
        for suffix in ['.json', '_joint_angles.json', '_vel.json']:
            json_file_path = os.path.join(directory, f"{new_base_name}{suffix}")
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as file:
                    data = json.load(file)
                    data['id'] = new_id
                    if suffix == '.json':
                        data['image_rgb'] = f"{new_base_name}.jpg"
                with open(json_file_path, 'w') as file:
                    json.dump(data, file, indent=4)

# Directory containing the images and JSON files
directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_clean_dup/'  # Replace with the path to your directory
remove_duplicates_and_rename(directory)
print("Duplicate files removed and remaining files renamed successfully.")