import os
import json
import cv2
import numpy as np

# def create_keypoints_video(images_dir, output_video_path, keypoint_indices, frame_rate=1000):
#     """
#     Display specified keypoints on images and create a video.
    
#     Args:
#         images_dir (str): Directory containing the images and JSON files.
#         output_video_path (str): Path to save the generated video.
#         keypoint_indices (list): List of keypoint indices to display.
#         frame_rate (int): Frame rate for the video.
#     """
#     # Get all image and JSON file paths
#     image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
#     json_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.json') and not f.endswith('_joint_angles.json') and not f.endswith('_vel.json')])
    
#     if len(image_files) != len(json_files):
#         raise ValueError("Mismatch in number of images and JSON files.")
    
#     # Initialize variables
#     output_images = []
#     first_image = cv2.imread(os.path.join(images_dir, image_files[0]))
#     height, width, _ = first_image.shape
    
#     # Define video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
#     video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
#     for image_file, json_file in zip(image_files, json_files):
#         # Load image
#         image_path = os.path.join(images_dir, image_file)
#         image = cv2.imread(image_path)
        
#         # Load JSON file
#         json_path = os.path.join(images_dir, json_file)
#         with open(json_path, 'r') as f:
#             data = json.load(f)
        
#         # Extract specified keypoints
#         for idx in keypoint_indices:
#             if idx < len(data['keypoints']):
#                 keypoint = data['keypoints'][idx][0]
#                 x, y, visibility = keypoint
#                 if visibility == 1:  # Only draw visible keypoints
#                     cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red dot
        
#         # Add the processed image to the video
#         video_writer.write(image)
    
#     # Release video writer
#     video_writer.release()
#     print(f"Video saved at: {output_video_path}")

# # Directory containing images and JSON files
# images_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/path_planning_rearranged_og/'

# # Output video file
# output_video_path = '/media/jc-merlab/Crucial X9/paper_data/data_collection_video_v2.mp4'

# # Indices of keypoints to display
# keypoint_indices = [3, 4, 6, 7, 8]

# Create video
# create_keypoints_video(images_dir, output_video_path, keypoint_indices)

def create_keypoints_video(images_dir, output_video_path, keypoint_indices, frame_rate=15):
    """
    Display specified keypoints on every 10th image, draw lines between them, and create a video.
    
    Args:
        images_dir (str): Directory containing the images and JSON files.
        output_video_path (str): Path to save the generated video.
        keypoint_indices (list): List of keypoint indices to display.
        frame_rate (int): Frame rate for the video.
    """
    # Get all image and JSON file paths
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    json_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.json') and not f.endswith('_joint_angles.json') and not f.endswith('_vel.json')])
    
    if len(image_files) != len(json_files):
        raise ValueError("Mismatch in number of images and JSON files.")
    
    # Initialize variables
    first_image = cv2.imread(os.path.join(images_dir, image_files[0]))
    height, width, _ = first_image.shape
    
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    # Process every 10th image
    for i in range(0, len(image_files), 10):  # Step every 10 images
        image_file = image_files[i]
        json_file = json_files[i]
        
        # Load image
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        
        # Load JSON file
        json_path = os.path.join(images_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract and draw keypoints and lines
        previous_point = None
        for idx in keypoint_indices:
            if idx < len(data['keypoints']):
                keypoint = data['keypoints'][idx][0]
                x, y, visibility = keypoint
                if visibility == 1:  # Only draw visible keypoints
                    current_point = (int(x), int(y))
                    cv2.circle(image, current_point, 8, (0, 255, 0), -1)  # Green dot
                    
                    # Draw line to the previous keypoint if it exists
                    if previous_point:
                        cv2.line(image, previous_point, current_point, (255, 0, 0), 4)  # Blue line
                    
                    previous_point = current_point
        
        # Add the processed image to the video
        video_writer.write(image)
    
    # Release video writer
    video_writer.release()
    print(f"Video saved at: {output_video_path}")

# Directory containing images and JSON files
images_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/3_kp/'

# Output video file
output_video_path = '/media/jc-merlab/Crucial X9/paper_data/data_collection_video_v4.mp4'

# Indices of keypoints to display
keypoint_indices = [3, 4, 6, 7, 8]

# Create video
create_keypoints_video(images_dir, output_video_path, keypoint_indices)
