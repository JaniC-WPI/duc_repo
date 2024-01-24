#!/usr/bin/env python3

import json
import os

# Directory containing your JSON files
directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/vel_reg_sim_test_2/'

# Function to convert floating points to integers in a list
def convert_to_int(lst):
    return [[int(coordinate) for coordinate in point] for point in lst]

# Iterate through each JSON file in the folder
for filename in sorted(os.listdir(directory)):
    if filename.endswith('_combined.json'):
        filepath = os.path.join(directory, filename)

        # Read the content of the file
        with open(filepath, 'r') as file:
            data = json.load(file)

        # Convert start_kp and next_kp values to integers
        for i in range(len(data['start_kp'])):
            data['start_kp'][i] = convert_to_int(data['start_kp'][i])
        for i in range(len(data['next_kp'])):
            data['next_kp'][i] = convert_to_int(data['next_kp'][i])

        # Write the updated content back to the file
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)

print("Conversion complete.")
