import cv2
import os

 

def create_video(image_folder, video_name, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    # Sort the images by name
    images.sort()

 

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

 

    # Define the codec using VideoWriter_fourcc() and create a VideoWriter object
    # We specify output file name video_name, FourCC code, FPS, and frame size (width x height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Or use 'MJPG' for .avi
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

 

    # Add frames to the video
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

 

    # Deallocate memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated

 

# DL prediction image data
folder = 8
pred = 1
file_path = f'/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/dl_prediction_result/{folder}/{pred}/'
# The function is called here
create_video(file_path, file_path + 'output_video.avi', 30)
