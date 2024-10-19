import cv2
import os

# Parameters
image_folder = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/ycb_test/'
video_name = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/ycb_test/ycb_test_01.avi'
frame_rate = 15  # Frames per second
extra_frames_for_last_image = 3  # Number of extra times to add the last image

images = []
for i in range(179):
    image_path = os.path.join(image_folder, f'zfill{i}.jpg')
    if os.path.exists(image_path):
        images.append(image_path)
    else:
        print(f"File not found: {image_path}")

if not images:
    print("No images found to create a video.")
    exit()

# Create VideoWriter object
frame = cv2.imread(images[0])
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))

# Write images to video
for image in images:
    video.write(cv2.imread(image))

# Add the last image multiple times
last_image = cv2.imread(images[-1])
for _ in range(extra_frames_for_last_image):
    video.write(last_image)

video.release()
cv2.destroyAllWindows()