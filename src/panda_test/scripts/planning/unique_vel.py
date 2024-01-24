#!/usr/bin/env python3

import json
import os
from collections import defaultdict

# Directory containing your JSON files
input_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/vel_reg_sim_test/'
# Directory to store unique velocities
output_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/vel_reg_sim_test/unique_vel_2/'

# Check if output directory exists, if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Dictionary to store unique velocities
unique_velocities = defaultdict(list)

# Iterate through each JSON file in the folder
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith('_vel.json'):
        filepath = os.path.join(input_dir, filename)
        
        with open(filepath, 'r') as file:
            data = json.load(file)
            vel_tuple = tuple(data['velocity'])  # Convert to tuple for hashability

            # Append the ID to the list of IDs for this unique velocity
            unique_velocities[vel_tuple].append(data['id'])

# Create new JSON files for each unique velocity in the output directory
for i, (velocity, ids) in enumerate(unique_velocities.items()):
    new_filename = f"{i:06d}_unique_vel.json"
    new_filepath = os.path.join(output_dir, new_filename)

    with open(new_filepath, 'w') as new_file:
        json.dump({"ids": ids, "unique_velocity": velocity}, new_file, indent=4)

print(f"Created {len(unique_velocities)} unique velocity files in {output_dir}.")