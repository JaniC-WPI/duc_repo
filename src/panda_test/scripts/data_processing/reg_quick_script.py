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
        # velocity_file = current_file.replace('.json', '_vel.json')

        joint_angles_file_1 = current_file.replace('.json', '_joint_angles.json')
        joint_angles_file_2 = next_file.replace('.json', '_joint_angles.json')

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

        # Read the joint angles from the corresponding files
        with open(os.path.join(data_dir, joint_angles_file_1), 'r') as file:
            joint_angles_data_1 = json.load(file)
        with open(os.path.join(data_dir, joint_angles_file_2), 'r') as file:
            joint_angles_data_2 = json.load(file)

        # Extract the joint angles lists
        joint_angles_1 = joint_angles_data_1['joint_angles']
        joint_angles_2 = joint_angles_data_2['joint_angles']

        # Calculate actual joint displacement
        joint_displacement = [j2 - j1 for j1, j2 in zip(joint_angles_1, joint_angles_2)]


        # Combine the data
        combined_data = {
            "start_kp": start_kp,
            "next_kp": next_kp,
            "position": position,
            "actual_joint_displacement": joint_displacement
        }

        # Save the combined data to a new JSON file
        combined_filename = current_file.replace('.json', '_combined.json')
        with open(os.path.join(out_dir, combined_filename), 'w') as outfile:
            json.dump(combined_data, outfile, indent=4)

        print(f"Combined data saved in {combined_filename}")

import os
import json
import random

def append_all_combined_json_to_original_set(data_dir, combine_number, repetitions, last_original_file_num):
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

import os
import json
import random

def corrected_combine_json_files(folder_path, combination_intervals, joint_angles_dir):
    # Get a list of all JSON files in the folder, sorted by file number
    json_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith("_combined.json")],
        key=lambda x: int(x.split('_')[0])
    )

    # Initialize the file index for new combined files
    last_file_index = int(json_files[-1].split('_')[0])
    
    # Iterate over starting indices and combination intervals
    for start_index in range(len(json_files)):
        for interval in combination_intervals:
            # Ensure the combination does not exceed the number of files
            if start_index + interval > len(json_files):
                continue            

            # Files to combine starting from the current index with the current interval
            files_to_combine = json_files[start_index:start_index + interval]

            print("files to combine", files_to_combine)

            # Initialize combined data structure
            combined_data = {
                "start_kp": None,
                "next_kp": None,
                "position": [0.0, 0.0, 0.0],  # Start with zeros for summing positions
                "actual_joint_displacement": None 
            }

            for index, file_name in enumerate(files_to_combine):
                file_path = os.path.join(folder_path, file_name)

                # Load JSON data from file
                with open(file_path, 'r') as file:
                    data = json.load(file)

                # Set start_kp from the first file and next_kp from the last file
                if index == 0:
                    combined_data["start_kp"] = data["start_kp"]
                if index == len(files_to_combine) - 1:
                    combined_data["next_kp"] = data["next_kp"]

                # Sum the position values
                combined_data["position"] = [sum(x) for x in zip(combined_data["position"], data["position"])]

            # Load joint angles to calculate the actual joint displacement
            first_joint_angles_file = f"{files_to_combine[0].split('_')[0]}_joint_angles.json"
            penultimate_joint_angles_file = f"{files_to_combine[-1].split('_')[0]}_joint_angles.json"
            base_name = penultimate_joint_angles_file.split('_joint_angles.json')[0]
            # Convert it to an integer and increment by 1
            last_number = int(base_name.lstrip('0')) + 1
            last_joint_angles_file = f"{str(last_number).zfill(6)}_joint_angles.json"
            print("last to last joint to combine", penultimate_joint_angles_file)
            first_joint_angles_path = os.path.join(joint_angles_dir, first_joint_angles_file)
            last_joint_angles_path = os.path.join(joint_angles_dir, last_joint_angles_file)

            print(os.path.exists(last_joint_angles_path))

            # if not os.path.exists(last_joint_angles_path):
            #     penultimate_joint_angles_file = f"{files_to_combine[-2].split('_')[0]}_joint_angles.json"
            # #     last_joint_angles_file = f"{files_to_combine[-2].split('_')[0]}_joint_angles.json"

            #     last_joint_angles_path = os.path.join(joint_angles_dir, last_joint_angles_file)

            print("First joint to combine", first_joint_angles_file)
            print("Last joint to combine", last_joint_angles_file)

            # Read the joint angles from the first and last files
            with open(first_joint_angles_path, 'r') as file:
                joint_angles_data_1 = json.load(file)
            with open(last_joint_angles_path, 'r') as file:
                joint_angles_data_2 = json.load(file)

            # Extract the joint angles lists
            joint_angles_1 = joint_angles_data_1['joint_angles']
            joint_angles_2 = joint_angles_data_2['joint_angles']

            # Calculate the actual joint displacement
            joint_displacement = [j2 - j1 for j1, j2 in zip(joint_angles_1, joint_angles_2)]

            # Set the calculated actual joint displacement in the combined data
            combined_data["actual_joint_displacement"] = joint_displacement

            # Update the file index for new file name
            last_file_index += 1
            new_file_name = f"{last_file_index:06}_combined.json"
            new_file_path = os.path.join(folder_path, new_file_name)

            # Write the combined data to a new JSON file
            with open(new_file_path, 'w') as new_file:
                json.dump(combined_data, new_file, indent=4)

            print(f"Created new combined file: {new_file_name} starting from index {start_index} with interval {interval}")

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

def is_valid_data(data, file_name):
    # Check if 'start_kp' and 'next_kp' each have 9 keypoints
    if len(data.get('start_kp', [])) != 9:
        print(f"Discarding {file_name}: 'start_kp' does not have 9 keypoints.")
        return False
    if len(data.get('next_kp', [])) != 9:
        print(f"Discarding {file_name}: 'next_kp' does not have 9 keypoints.")
        return False
    
    # Check if actual_joint_displacement is not all zero
    actual_joint_displacement = data.get('actual_joint_displacement', [])
    if actual_joint_displacement and all(abs(i) < 1e-10 for i in actual_joint_displacement):
        print(f"Discarding {file_name}: 'actual_joint_displacement' is all zeros.")
        return False
    
    # Check if position is not all zero
    position = data.get('position', [])
    if position and all(abs(i) < 1e-10 for i in position):
        print(f"Discarding {file_name}: 'position' is all zeros.")
        return False
    
    return True

def process_json_files(folder):
    valid_files = []
    invalid_folder = os.path.join(folder, "invalid_files")
    os.makedirs(invalid_folder, exist_ok=True)
    
    # Get list of all files in the folder
    for file_name in sorted(os.listdir(folder)):
        if file_name.endswith("_combined.json"):
            file_path = os.path.join(folder, file_name)
            try:
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                
                # Check if the data is valid
                if is_valid_data(data, file_name):
                    valid_files.append(file_name)
                else:
                    # Move invalid file to invalid folder
                    print(f"Moving {file_name} to invalid_files folder.")
                    shutil.move(file_path, os.path.join(invalid_folder, file_name))
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
    
    # Renumber and rename valid files
    for idx, valid_file in enumerate(valid_files):
        old_path = os.path.join(folder, valid_file)
        new_name = f"{str(idx).zfill(6)}_combined.json"
        new_path = os.path.join(folder, new_name)
        print(f"Renaming {valid_file} to {new_name}")
        os.rename(old_path, new_path)
    
    # Remove the invalid_files folder if it exists and contains files
    if os.path.exists(invalid_folder):
        if not os.listdir(invalid_folder):
            os.rmdir(invalid_folder)  # Remove only if it's empty
            print("Invalid folder was empty and removed.")
        else:
            shutil.rmtree(invalid_folder)  # Force remove folder and its contents
            print("Invalid folder and its contents removed.")

if __name__ == "__main__":
    # Load configurations from JSON files
    # directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/path_plan_kp_phys/'
    # out_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/path_plan_kp_phys_combined/'
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/20_kp/'
    out_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/20_all_out/'
    combination_intervals = [10, 20, 30, 40, 50, 60]
    # combination_intervals = [5]

    # combine_json_with_velocity(directory, out_dir)
    # process_json_files(out_dir)
    # corrected_combine_json_files(out_dir, combination_intervals, directory)

    # update_velocity_json(directory)
    
    # append_all_combined_json_to_original_set(out_dir, 100, 1000, 158)
    # clean_and_renumber_json_files(out_dir)

    # Define your source folders and destination folder
    source_folders = [
                    '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/1_all_out/', 
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/2_all_out/', 
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/3_all_out/', 
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/4_all_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/5_all_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/6_all_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/7_all_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/8_all_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/9_all_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/10_all_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/11_all_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/12_all_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/13_all_out/',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/14_all_out',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/15_all_out',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/16_all_out',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/17_all_out',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/18_all_out',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/19_all_out',
                      '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/20_all_out']

    destination_folder = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/regression_rearranged_all_corrected/'
    combine_and_renumber_folders(source_folders, destination_folder)



