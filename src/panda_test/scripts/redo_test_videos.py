import cv2
import glob
import os

# Define the folder containing your images
folder = '/media/jc-merlab/Crucial X9/ros_direct_inpaint/occlusion_img_op/exp_02/'
# Adjust the pattern if your images have a different extension (e.g., '*.jpg')
image_paths = sorted(glob.glob(os.path.join(folder, '*.jpg')))

if not image_paths:
    raise FileNotFoundError("No images found in the folder!")

# Read the first image to get frame dimensions
frame = cv2.imread(image_paths[0])
if frame is None:
    raise ValueError("Could not read the first image. Please check the file format.")
height, width, channels = frame.shape

# Verify the expected dimensions (optional)
if (width, height) != (640, 480):
    print(f"Warning: Expected dimensions 640x480, but got {width}x{height}")

# Setup the VideoWriter
output_video_path = '/media/jc-merlab/Crucial X9/ros_direct_inpaint/ros_inpaint_02.avi'
# Define the codec and create VideoWriter object.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 15 # Adjust FPS as needed
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Couldn't read {img_path}. Skipping.")
        continue

    # Black out the left border (columns 0 to 19) and right border (columns 620 to 639)
    img[:, 0:20] = 0         # Left border: all rows, columns 0-19
    img[:, 600:640] = 0       # Right border: all rows, columns 620-639

    # Write the modified frame to the video
    video_writer.write(img)

# Release the video writer
video_writer.release()
print(f"Video saved as {output_video_path}")