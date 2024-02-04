# import cv2
# import os

# # Parameters
# image_folder = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/exps/1/'  # Replace with the path to your folder
# video_name = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/exps/1/output_video.avi'
# frame_rate = 30  # frames per second

# images = [f"{i}.png" for i in range(270)]  # List of images from 0.png to 213.png
# images.append('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/exps/1/sim_published_goal_image.jpg')  # Add the last image

# # Determine the width and height from the first image
# image_path = os.path.join(image_folder, images[0])
# frame = cv2.imread(image_path)
# height, width, layers = frame.shape

# # Initialize video writer
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs as well.
# video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

# # Add images to video
# for image in images:
#     image_path = os.path.join(image_folder, image)
#     if os.path.exists(image_path):  # Check if the image file exists
#         video.write(cv2.imread(image_path))
#     else:
#         print(f"File {image_path} not found")

# video.release()  # Finalize the video file.

import cv2
import os

# Parameters
image_folder = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/exps/2/'  # Replace with your folder path
video_name = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/exps/2/output_video.avi'
frame_rate = 30  # frames per second
# overlay_image_path = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/exps/2/sim_published_goal_image.jpg'  # Path to overlay image
extend_last_frame_count = 4  # Number of times to repeat the last frame

# Read the overlay image
# overlay = cv2.imread(overlay_image_path)

# Generate image list
images = [f"{i}.png" for i in range(1149)]  # Adjust the range if needed

# Determine the width and height from the first image
image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(image_path)
height, width, layers = frame.shape

# Resize overlay image to match video frame size
# overlay_resized = cv2.resize(overlay, (width, height))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs as well.
video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

# Alpha weights for blending
alpha = 0.6  # Adjust as needed
beta = (1.0 - alpha)

# Add images to video with overlay
for image in images:
    image_path = os.path.join(image_folder, image)
    if os.path.exists(image_path):
        frame = cv2.imread(image_path)
        # blended_frame = cv2.addWeighted(frame, alpha, overlay_resized, beta, 0)
        # video.write(blended_frame)
        video.write(frame)
    else:
        print(f"File {image_path} not found")

# Extend the last frame
last_frame = frame  # The last frame processed
for _ in range(extend_last_frame_count):
    video.write(last_frame)

video.release()  # Finalize the video file.