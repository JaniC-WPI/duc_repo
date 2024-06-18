import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

folder_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged/'

# Initialize lists to hold joint angles
j1_angles = []
j2_angles = []
j3_angles = []

# Iterate over each file in the folder
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith('_joint_angles.json'):  # Ensures we only read .json files
        # file_path = os.path.join(folder_path, filename) # uncomment when all the configurations are considered
        # Check if the file index is greater than or equal to 010000

        #Uncomment below when starting from a particlar index
        file_index = int(filename.split('_')[0])
        print(file_index)
        if file_index >= 10000:
            file_path = os.path.join(folder_path, filename)
        
            # Open and read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                joint_angles = data['joint_angles']
                
                # Append each angle to its respective list
                j1_angles.append(joint_angles[0])
                j2_angles.append(joint_angles[1])
                j3_angles.append(joint_angles[2])

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting all configurations as points in 3D space
ax.scatter(j1_angles, j2_angles, j3_angles, c='g', marker='_')

# Setting labels for axes
ax.set_xlabel('Joint 1 angle')
ax.set_ylabel('Joint 2 angle')
ax.set_zlabel('Joint 3 angle')

# Display the plot
plt.show()