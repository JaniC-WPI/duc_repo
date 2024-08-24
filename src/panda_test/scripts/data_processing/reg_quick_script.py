import os
import json

def update_velocity_json(folder_path):
    """
    Updates all *_vel.json files in the specified folder by adding a "time_rate" key with a value of 1/30.

    Parameters:
    - folder_path: Path to the folder containing the JSON files.
    """
    # Iterate over each file in the directory
    for file_name in sorted(os.listdir(folder_path)):
        # Check if the file is a velocity JSON file
        if file_name.endswith('_vel.json'):
            # Construct the full path to the file
            full_path = os.path.join(folder_path, file_name)
            
            # Open and read the JSON file
            with open(full_path, 'r') as file:
                data = json.load(file)
            
            # Add the 'time_rate' key with the value of 1/30
            data['time_rate'] = 1/30
            
            # Write the modified data back to the file
            with open(full_path, 'w') as file:
                json.dump(data, file, indent=4)

    print("Update complete.")

import os
import json

def combine_json_with_velocity(data_dir, out_dir):
    # Helper function to sort file names numerically
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Get the list of JSON files (excluding velocity files)
    keypoint_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json') and not f.endswith('_vel.json') and not f.endswith('_combined.json') and not f.endswith('_joint_angles.json')])
    # Process each file (except the last one)
    for i in range(len(keypoint_files) - 1):
        current_file = keypoint_files[i]
        next_file = keypoint_files[i + 1]
        velocity_file = next_file.replace('.json', '_vel.json')

        # Read the current keypoints JSON file
        with open(os.path.join(data_dir, current_file), 'r') as file:
            current_data = json.load(file)
        start_kp = current_data['keypoints']

        # Read the next keypoints JSON file
        with open(os.path.join(data_dir, next_file), 'r') as file:
            next_data = json.load(file)
        next_kp = next_data['keypoints']

        # Read the corresponding velocity JSON file
        with open(os.path.join(data_dir, velocity_file), 'r') as file:
            velocity_data = json.load(file)
        velocity = velocity_data['velocity']
        # time = velocity_data['time_rate']
        time = 1/15
        
        position = [v * time for v in velocity]

        # Combine the data
        combined_data = {
            "start_kp": start_kp,
            "next_kp": next_kp,
            "position": position
        }

        # Save the combined data to a new JSON file
        combined_filename = current_file.replace('.json', '_combined.json')
        with open(os.path.join(out_dir, combined_filename), 'w') as outfile:
            json.dump(combined_data, outfile, indent=4)

        print(f"Combined data saved in {combined_filename}")

import os
import json
import random

def append_new_combined_json_to_original_set(data_dir, combine_number, repetitions, last_original_file_num):
    original_combined_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith('_combined.json') and int(f.split('_')[0]) <= last_original_file_num]

    if len(original_combined_files) < 2:
        print("Not enough original combined files to process.")
        return

    for _ in range(repetitions):
        start_index = random.randint(0, len(original_combined_files) - 2)  # Avoid picking the last file
        # Calculate the maximum possible index for adding numbers
        max_add_index = len(original_combined_files) - start_index - 1

        # Ensure there is a valid range for add_number
        if max_add_index < 4:
            print("Not enough files remaining to process after start_index.")
            continue  # Skip to the next iteration of the loop
        # Ensure the add number is >3 and <201
        # add_number = random.randint(4, min(200, len(original_combined_files) - start_index - 1))

        add_number = random.randint(4, min(combine_number, max_add_index))
        
        end_index = min(start_index + add_number, len(original_combined_files) - 1)
        
        first_file = original_combined_files[start_index]
        second_file = original_combined_files[end_index]
        
        print(f"Picked files: {first_file} and {second_file}, with added number: {add_number}")
        
        with open(os.path.join(data_dir, first_file), 'r') as f:
            first_data = json.load(f)
        start_kp = first_data['start_kp']
        
        with open(os.path.join(data_dir, second_file), 'r') as f:
            second_data = json.load(f)
        next_kp = second_data['next_kp']
        
        cumulative_position = [0.0, 0.0, 0.0]
        for i in range(start_index, end_index + 1):
            with open(os.path.join(data_dir, original_combined_files[i]), 'r') as f:
                data = json.load(f)
            position = data['position']
            cumulative_position = [sum(x) for x in zip(cumulative_position, position)]
        
        # Determine the next file number to append by counting all files, including new ones
        all_combined_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_combined.json')])
        new_file_num = int(all_combined_files[-1].split('_')[0]) + 1 if all_combined_files else 0
        new_combined_filename = f'{new_file_num:06d}_combined.json'
        
        with open(os.path.join(data_dir, new_combined_filename), 'w') as outfile:
            json.dump({
                "start_kp": start_kp,
                "next_kp": next_kp,
                "position": cumulative_position
            }, outfile, indent=4)
        
        print(f"New combined data appended as {new_combined_filename}")

def clean_and_renumber_json_files(folder_path):
    # Step 1: Identify and remove invalid files
    valid_files = []
    for file_name in sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[0])):
        if file_name.endswith('_combined.json'):
            with open(os.path.join(folder_path, file_name), 'r') as file:
                try:
                    data = json.load(file)
                    # Check if any key has empty data
                    if data['position'] and data['start_kp'] and data['next_kp']:
                        valid_files.append(data)  # Save valid data for renumbering
                    else:
                        print(f"Removing {file_name} due to empty data fields.")
                except json.JSONDecodeError:
                    print(f"Error reading {file_name}. Skipping.")
    
    # Step 2: Remove all original files to avoid conflict during renumbering
    for file_name in os.listdir(folder_path):
        if file_name.endswith('_combined.json'):
            os.remove(os.path.join(folder_path, file_name))
    
    # Step 3: Renumber and save valid files
    for new_index, data in enumerate(valid_files):
        new_file_name = f"{new_index:06d}_combined.json"
        new_file_path = os.path.join(folder_path, new_file_name)
        with open(new_file_path, 'w') as new_file:
            json.dump(data, new_file, indent=4)
        print(f"Saved {new_file_name}")

import os
import shutil
import json

def combine_and_renumber_folders(source_folders, dest_folder):
    """
    Combine and renumber JSON files from multiple source folders into a single destination folder,
    ensuring sequential numbering.
    
    Parameters:
    - source_folders: List of paths to the source folders.
    - dest_folder: Path to the destination folder.
    """
    # Ensure the destination folder exists
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    all_files = []
    # Collect all files from each source folder
    for folder in source_folders:
        folder_files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('_combined.json')]
        all_files.extend(folder_files)
    
    # Renumber and copy files to the destination folder
    for i, file_path in enumerate(all_files, start=0):  # Start enumeration from 0
        new_file_name = f"{i:06d}_combined.json"
        new_file_path = os.path.join(dest_folder, new_file_name)
        
        shutil.copy(file_path, new_file_path)
        
        print(f"Copied {file_path} to {new_file_path}")

    


# def combine_json_with_velocity(folder_path, output_path):
#     # Helper function to sort file names numerically
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     def sort_numerically(file_name):
#         parts = file_name.split('.')[0]
#         base, number = parts[:-3], parts[-3:]
#         return int(number)

#     # Find all json files that don't end with '_vel.json'
#     json_files = [f for f in sorted(os.listdir(folder_path), key=sort_numerically) if f.endswith('.json') and not f.endswith('_vel.json')]
    
#     for i in range(len(json_files) - 1):
#         # Construct file names
#         start_file = os.path.join(folder_path, json_files[i])
#         next_file = os.path.join(folder_path, json_files[i + 1])
#         vel_file = os.path.join(folder_path, json_files[i].replace('.json', '_vel.json'))
        
#         # Read start, next, and velocity JSON files
#         with open(start_file, 'r') as f:
#             start_data = json.load(f)
#         with open(next_file, 'r') as f:
#             next_data = json.load(f)
#         with open(vel_file, 'r') as f:
#             vel_data = json.load(f)
        
#         # Calculate position (assuming velocity is a vector and time_rate is 1/30)
#         position = [v * (1/30) for v in vel_data.get('velocity', [])]
        
#         # Combine into a new structure
#         combined_data = {
#             'start_kp': start_data.get('keypoints', []),
#             'next_kp': next_data.get('keypoints', []),
#             'position': position
#         }
        
#         # Write combined data to a new JSON file
#         combined_file_name = os.path.join(output_path, f"{json_files[i].split('.')[0]}_combined.json")
#         with open(combined_file_name, 'w') as f:
#             json.dump(combined_data, f, indent=4)
    
#     print("Combination complete.")

if __name__ == "__main__":
    # Load configurations from JSON files
    # directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/path_plan_kp_phys/'
    # out_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/path_plan_kp_phys_combined/'
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/20_kp_cleaned/'
    out_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/20_half_half_out/'

    # update_velocity_json(directory)
    # combine_json_with_velocity(directory, out_dir)
    # append_new_combined_json_to_original_set(out_dir, 100, 1000, 158)
    # clean_and_renumber_json_files(out_dir)

    # Define your source folders and destination folder
    source_folders = [
                    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/1_half_out/', 
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/2_half_out/', 
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/3_half_out/', 
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/4_half_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/5_half_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/6_half_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/7_half_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/8_half_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/9_half_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/10_half_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/11_half_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/12_half_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/13_half_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/14_half_out',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/15_half_out',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/16_half_out',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/17_half_out',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/18_half_out',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/19_half_out',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/20_half_out']

    destination_folder = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/regression_rearranged_half/'
    combine_and_renumber_folders(source_folders, destination_folder)



