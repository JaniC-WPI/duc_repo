import os
import shutil

# Specify the source and destination folders
source_folder_path = '/media/jc-merlab/SanDisk/Data/path_planning_clean_dup/'
destination_folder_path = '/media/jc-merlab/SanDisk/Data/source_physical_occlusion/'

# Ensure the destination folder exists
os.makedirs(destination_folder_path, exist_ok=True)

# Iterate through all files in the source folder
for filename in os.listdir(source_folder_path):
    # Check if the file is a .jpg or .json file but not a _vel.json file
    if filename.endswith('.jpg') or (filename.endswith('.json') and not filename.endswith('_vel.json') and not filename.endswith('_joint_angles.json')):
        # Construct full file paths
        source_file_path = os.path.join(source_folder_path, filename)
        destination_file_path = os.path.join(destination_folder_path, filename)
        
        # Copy the file to the destination folder
        shutil.copyfile(source_file_path, destination_file_path)

print("Files copied successfully!")

# import os
# import json
# import time
# import sys

# folder_path = "/home/jc-merlab/Pictures/panda_data/panda_sim_vel/path_planning_panda_train/"

# # List all JSON files in the folder
# json_files = [file for file in sorted(os.listdir(folder_path)) if file.endswith('.json')]

# # # Redirect output
# # sys.stdout = open('output.txt', 'w')  

# with open('output.txt', 'w') as output_file:
#     # Iterate through each JSON file
#     for json_file in json_files:
#         file_path = os.path.join(folder_path, json_file)

#         # Load JSON data from the file
#         with open(file_path, 'r') as json_data_files:
#             data = json.load(json_data_files)

#         # Check if "bboxes" and "keypoints" keys exist in the loaded data
#         if "bboxes" in data and "keypoints" in data:
#             output_file.write(f"{json_file}: 'bboxes' and 'keypoints' keys found\n")
#             print(f"{json_file}: 'bboxes' and 'keypoints' keys found")
#             # time.sleep(1)
#         elif "bboxes" in data:
#             output_file.write(f"{json_file}: 'bboxes' key found, but 'keypoints' key missing\n")
#             print(f"{json_file}: 'bboxes' key found, but 'keypoints' key missing")
#             # time.sleep(1)
#         elif "keypoints" in data:
#             output_file.write(f"{json_file}: 'keypoints' key found, but 'bboxes' key missing\n")
#             print(f"{json_file}: 'keypoints' key found, but 'bboxes' key missing")
#             # time.sleep(1)
#         else:
#             output_file.write(f"{json_file}: 'bboxes' and 'keypoints' keys missing\n")
#             print(f"{json_file}: 'bboxes' and 'keypoints' keys missing")
#             # time.sleep(2)

# import os
# import json

# # Path to the folder containing your JSON files
# folder_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/path_planning_panda_valid/'

# # Loop through all files in the folder
# for file_name in sorted(os.listdir(folder_path)):
#     # Check if the file is a JSON file
#     if file_name.endswith('.json') and not file_name.endswith('_vel.json'):
#         # Construct the full file path
#         file_path = os.path.join(folder_path, file_name)
        
#         # Open the JSON file for reading
#         with open(file_path, 'r') as file:
#             data = json.load(file)
        
#         # Add the 'valid' key with a value of 0 (considering all as invalid initially)
#         data['valid'] = 1
        
#         # Open the JSON file for writing
#         with open(file_path, 'w') as file:
#             # Write the modified data back to the file
#             json.dump(data, file, indent=4)

# print("All files have been updated with a 'valid' key.")

    
