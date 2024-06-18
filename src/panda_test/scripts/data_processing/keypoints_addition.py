import json
import os
import numpy as np

def interpolate_points(point_a, point_b):
    return [(point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2, 1]

def extend_and_drop_perpendicular(point_a, point_b):
    """
    Computes a perpendicular from point_b (6th keypoint) using the direction and length of the vector from point_a to point_b.
    The perpendicular will be of the same length as the vector from point_a to point_b.
    """
    vector = np.array([point_b[0] - point_a[0], point_b[1] - point_a[1]])
    length = np.linalg.norm(vector)
    perpendicular_vector = np.array([-vector[1], vector[0]])  # Rotate 90 degrees to get the perpendicular
    final_point = np.array([point_b[0], point_b[1]]) + perpendicular_vector / np.linalg.norm(perpendicular_vector) * (length * 1.25)
    return [final_point[0], final_point[1], 1] 

def add_bounding_box_around_keypoint(keypoint, size):
    half_size = size / 2
    return [keypoint[0] - half_size, keypoint[1] - half_size,
            keypoint[0] + half_size, keypoint[1] + half_size]

def update_keypoints(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Interpolate between 4th and 5th keypoints
    point_a = data['keypoints'][3][0]
    point_b = data['keypoints'][4][0]
    new_keypoint1 = interpolate_points(point_a, point_b)

    # Interpolate between 2nd and 3rd keypoints
    point_a = data['keypoints'][1][0]
    point_b = data['keypoints'][2][0]
    new_keypoint2 = interpolate_points(point_a, point_b)

    # Extend and drop perpendicular from 5th to 6th keypoints
    point_c = data['keypoints'][4][0]
    point_d = data['keypoints'][5][0]
    new_keypoint3 = extend_and_drop_perpendicular(point_c, point_d)

    # Calculate the bounding boxes
    new_bbox1 = add_bounding_box_around_keypoint(new_keypoint1, size=20)
    new_bbox2 = add_bounding_box_around_keypoint(new_keypoint2, size=20)
    new_bbox3 = add_bounding_box_around_keypoint(new_keypoint3, size=20)

    # Insert the new keypoints and bounding boxes
    data['keypoints'].insert(4, [new_keypoint1])
    data['bboxes'].insert(4, new_bbox1)
    data['keypoints'].insert(2, [new_keypoint2])
    data['bboxes'].insert(2, new_bbox2)
    data['keypoints'].insert(8, [new_keypoint3])  # Insert after the interpolated keypoint
    data['bboxes'].insert(8, new_bbox3)

    # Save the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def process_all_json_files(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json') and not file_name.endswith('_combined.json') and not file_name.endswith('_vel.json') and not file_name.endswith('_joint_angles.json'):
            file_path = os.path.join(folder_path, file_name)
            update_keypoints(file_path)
            print(f"Processed {file_name}")

folder_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/path_plan_kp_sim_v3/'
process_all_json_files(folder_path)