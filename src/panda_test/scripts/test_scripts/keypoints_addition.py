import json
import os

def interpolate_points(point_a, point_b):
    """
    Performs linear interpolation between two points.
    Since it's a simple midpoint calculation here, it's technically the same.
    But you can adjust the logic here for different types of interpolation if needed.
    """
    return [(point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2, 1]

def add_bounding_box_around_keypoint(keypoint, size):
    """
    Creates a bounding box of a given size around a keypoint.
    The size is applied equally in all directions from the keypoint's center.
    """
    half_size = size / 2
    return [keypoint[0] - half_size, keypoint[1] - half_size,
            keypoint[0] + half_size, keypoint[1] + half_size]

def update_keypoints(file_path):
    """
    Updates the keypoints by adding an interpolated keypoint between the 4th and 5th keypoints.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Assuming 'keypoints' structure is consistent with the provided example
    point_a = data['keypoints'][3][0]
    point_b = data['keypoints'][4][0]
    new_keypoint = interpolate_points(point_a, point_b)

    # Calculate the bounding box around the new keypoint
    new_bbox = add_bounding_box_around_keypoint(new_keypoint, size=20)
    
    # Insert the new keypoint
    data['keypoints'].insert(4, [new_keypoint])
    data['bboxes'].insert(4, new_bbox)
    
    # Save the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def process_all_json_files(folder_path):
    """
    Process all JSON files in the given folder.
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json') and not file_name.endswith('_combined.json') and not file_name.endswith('_vel.json'):
            file_path = os.path.join(folder_path, file_name)
            update_keypoints(file_path)
            print(f"Processed {file_name}")

# Specify the folder containing the JSON files
folder_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/path_plan_kp_sim/'

# Process all JSON files in the folder
process_all_json_files(folder_path)