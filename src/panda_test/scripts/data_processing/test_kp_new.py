import json
import os

''' 
This file was written to test if all keypoints are correctly getting saved. If there are keypoints more than the length specified in the the function validate_keypoints, 
then any file with keypoints other than the length will be deemed invalid. This is a debug node
'''

def validate_keypoints(keypoints, expected_length=9):
    """ Validate if keypoints list has exactly the expected number of keypoints,
        and each keypoint has exactly 2 coordinates (x and y).
    """
    return len(keypoints) == expected_length and all(len(kp[0]) == 3 for kp in keypoints)

def check_json_files(directory):
    invalid_files = []  # To store names of files with invalid data

    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('_combined.json'): # and not filename.endswith('_joint_angles.json') and not filename.endswith('_vel.json'):  # Check only JSON files of interest
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)

                start_kp = data.get('start_kp')
                next_kp = data.get('next_kp')
                # keypoints = data.get('keypoints')
                # Check both start_kp and next_kp for the correct number of keypoints
                if not (validate_keypoints(start_kp) and validate_keypoints(next_kp)):
                    invalid_files.append(filename)  # Log the invalid file name

    # Print or log the results
    if invalid_files:
        print(f"Found {len(invalid_files)} files with invalid keypoint structures:")
        for file in invalid_files:
            print(file)
    else:
        print("All files have the correct keypoint structure.")

# Set the directory path to check
directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/regression_rearranged/'
check_json_files(directory)