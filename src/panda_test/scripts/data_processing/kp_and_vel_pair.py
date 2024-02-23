import json
import os

# Directory containing your JSON files
data_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/path_planning_panda_regression/'

# Get the list of JSON files (excluding velocity files)
keypoint_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json') and not f.endswith('_vel.json')])

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
    time = velocity_data['time_rate']

    # Combine the data
    combined_data = {
        "start_kp": start_kp,
        "next_kp": next_kp,
        "velocity": velocity
    }

    # Save the combined data to a new JSON file
    combined_filename = current_file.replace('.json', '_combined.json')
    with open(os.path.join(data_dir, combined_filename), 'w') as outfile:
        json.dump(combined_data, outfile, indent=4)

    print(f"Combined data saved in {combined_filename}")