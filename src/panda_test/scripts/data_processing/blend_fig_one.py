# import cv2
# import os


# # image_paths = cv2.imread('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/no_obs/nn_25_astar_custom_old/2/path/path_4.jpg')

# # image_1 = cv2.imread('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_no_obs/2/raw/76.png')

# # image_2 = cv2.imread('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_no_obs/2/raw/117.png')

# # image_3 = cv2.imread('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_no_obs/2/raw/174.png')

# # image_4 = cv2.imread('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_no_obs/2/raw/299.png')

# # image_5 = cv2.imread('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_no_obs/2/raw/556.png')

# # image_6 = cv2.imread('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/custom/astar_latest_no_obs/2/raw/1309.png')

# # goal_image_1 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/paper_data/figure_1_images/fig_1_a.png'
# # goal_image_2 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/paper_data/figure_1_images/fig_1_b.png'
# # goal_image_3 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/paper_data/figure_1_images/fig_1_c.png'
# # goal_image_4 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/paper_data/figure_1_images/fig_1_d.png'
# # goal_image_5 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/paper_data/figure_1_images/fig_1_e.png'
# # goal_image_6 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/paper_data/figure_1_images/fig_1_f.png'

# # blend_1 = cv2.addWeighted(image_1, 0.8, image_paths, 1.2, 0)
# # cv2.imwrite(goal_image_1, blend_1)

# # blend_2 = cv2.addWeighted(image_2, 0.8, image_paths, 1.2, 0)
# # cv2.imwrite(goal_image_2, blend_2)

# # blend_3 = cv2.addWeighted(image_3, 0.8, image_paths, 1.2, 0)
# # cv2.imwrite(goal_image_3, blend_3)

# # blend_4 = cv2.addWeighted(image_4, 0.8, image_paths, 1.2, 0)
# # cv2.imwrite(goal_image_4, blend_4)

# # blend_5 = cv2.addWeighted(image_5, 0.8, image_paths, 1.2, 0)
# # cv2.imwrite(goal_image_5, blend_5)

# # blend_6 = cv2.addWeighted(image_6, 0.8, image_paths, 1.2, 0)
# # cv2.imwrite(goal_image_6, blend_6)


# import os
# import json

# # Path to your folder containing the JSON and JPEG files
# data_folder = '/media/jc-merlab/Crucial X9/occ_new_panda_physical_dataset/'

# # Function to check the number of keypoints in a JSON file
# def check_keypoints(json_file):
#     with open(json_file, 'r') as f:
#         data = json.load(f)
#         keypoints = data.get('keypoints', [])
#         if len(keypoints) < 9:
#             print(f"File {json_file} has less than 9 keypoints. Found {len(keypoints)} keypoints.")
#         else:
#             print(f"File {json_file} has all 9 keypoints.")

# # Iterate through all files in the folder
# def check_data_folder(data_folder):
#     for root, dirs, files in os.walk(data_folder):
#         for file in files:
#             if file.endswith('.json'):  # Only process JSON files
#                 json_file_path = os.path.join(root, file)
#                 check_keypoints(json_file_path)

# # Run the script
# check_data_folder(data_folder)

import os
import shutil

def extract_files(src_folder, dest_folder, num_files=50000):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # List all files in the source folder
    files = sorted(os.listdir(src_folder))

    # Filter JSON and corresponding JPG files
    json_files = [f for f in files if f.endswith('.json')]
    json_files = json_files[:num_files]  # Get the first 50,000 JSON files

    for json_file in json_files:
        base_name = json_file[:-5]  # Remove the .json extension
        jpg_file = base_name + '.jpg'

        # Construct full paths
        json_src_path = os.path.join(src_folder, json_file)
        jpg_src_path = os.path.join(src_folder, jpg_file)
        
        json_dest_path = os.path.join(dest_folder, json_file)
        jpg_dest_path = os.path.join(dest_folder, jpg_file)

        # Copy JSON file
        if os.path.exists(json_src_path):
            shutil.copy2(json_src_path, json_dest_path)

        # Copy corresponding JPG file
        if os.path.exists(jpg_src_path):
            shutil.copy2(jpg_src_path, jpg_dest_path)

    print(f"Successfully copied {num_files} JSON and corresponding JPG files to {dest_folder}")

# Set source and destination folders
src_folder = '/media/jc-merlab/Crucial X9/occ_new_panda_physical_dataset'  # Replace with your source folder path
dest_folder = '/media/jc-merlab/Crucial X9/occ_new_panda_physical_dataset_trunc'  # Replace with your destination folder path

# Extract first 50,000 files
extract_files(src_folder, dest_folder, num_files=50000)