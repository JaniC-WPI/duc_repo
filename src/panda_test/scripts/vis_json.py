import os
import json
import cv2
import glob

# Constants
SOURCE_FOLDER = '/home/jc-merlab/Pictures/panda_data/kprcnn_ur_dataset_json_2/'
DEST_FOLDER = '/home/jc-merlab/Pictures/panda_data/json_output/'
VIDEO_NAME = '/home/jc-merlab/Pictures/panda_data/json_output/output_video.avi'

# Ensure destination folder exists
if not os.path.exists(DEST_FOLDER):
    os.makedirs(DEST_FOLDER)

# Step 1: Loop through each JSON file
for json_path in sorted(glob.glob(os.path.join(SOURCE_FOLDER, '*.json'))):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Step 3: Load the image using the 'image_rgb' field
    image_path = os.path.join(SOURCE_FOLDER, data['image_rgb'])
    image = cv2.imread(image_path)

    # Step 4: Draw keypoints on the image
    for keypoints in data['keypoints']:
        for kp in keypoints:
            x, y, visibility = kp
            if visibility == 1:  # Check if the keypoint is visible
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Step 5: Save the modified image in a new folder
    dest_image_path = os.path.join(DEST_FOLDER, data['image_rgb'])
    cv2.imwrite(dest_image_path, image)

# Step 6: Convert images in the new folder into a video
images = sorted(glob.glob(os.path.join(DEST_FOLDER, '*.jpg')))
frame = cv2.imread(images[0])
h, w, l = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(VIDEO_NAME, fourcc, 10, (w, h))

for image_path in images:
    image = cv2.imread(image_path)
    out.write(image)

out.release()
print(f"Video saved as {VIDEO_NAME}")