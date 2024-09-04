import os
import json
import numpy as np

def find_goal_config_in_folder(goal_config, folder_path):
    """
    Searches for the goal configuration in JSON files within the given folder and retrieves the corresponding joint angles.

    Args:
    - goal_config (np.ndarray): The goal configuration to search for (array of keypoints).
    - folder_path (str): Path to the folder containing JSON files.

    Returns:
    - joint_angles (list): The joint angles corresponding to the found goal configuration.
    - file_number (str): The file number where the goal configuration was found.
    """
    # Convert goal_config to a list of tuples for easier comparison
    goal_config_list = [tuple(point) for point in goal_config]

    print(goal_config_list)

    # Traverse through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json') and not filename.endswith('_joint_angles.json') and not filename.endswith('_vel.json') :
            json_path = os.path.join(folder_path, filename)
            
            with open(json_path, 'r') as file:
                data = json.load(file)
                
                # Extract keypoints from the JSON file
                keypoints = data.get('keypoints', [])
                keypoints_list = [tuple(map(int, kp[0][:2])) for kp in keypoints]  # Convert to list of tuples (x, y)
                # print(keypoints_list)

                # Check if the goal_config matches the keypoints in the JSON file
                if keypoints_list == goal_config_list:
                    # If a match is found, extract the file number
                    file_number = filename.split('.')[0]
                    
                    # Construct the joint angles file path
                    joint_angles_filename = f"{file_number}_joint_angles.json"
                    joint_angles_path = os.path.join(folder_path, joint_angles_filename)

                    # Load joint angles from the corresponding _joint_angles.json file
                    with open(joint_angles_path, 'r') as joint_file:
                        joint_data = json.load(joint_file)
                        joint_angles = joint_data.get('joint_angles', [])

                    return joint_angles, file_number

    # If no match is found, return None
    return None, None

# Example usage
goal_config = np.array([[250, 442], [252, 311], [210, 271], [167, 231], [188, 209], [227, 147], [265, 85], [278, 56], [315, 73]])
folder_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged/'  # Replace with your folder path

joint_angles, file_number = find_goal_config_in_folder(goal_config, folder_path)

if joint_angles is not None:
    print(f"Joint Angles: {joint_angles}")
    print(f"File Number: {file_number}")
else:
    print("Goal configuration not found in any file.")