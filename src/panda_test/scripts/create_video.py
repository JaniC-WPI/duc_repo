import cv2
import os

def create_video(image_folder, video_name, fps):
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    # Sort the images by name
    # images.sort()

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

# The function is called here
create_video('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps/14/', '/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps/14/output_video.avi', 30)
