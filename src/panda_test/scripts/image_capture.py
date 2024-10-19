#!/usr/bin/env python3.8

import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge, CvBridgeError
import sys
import rospy
import os
from os.path import expanduser
import glob
import time

# Define CvBridge for ROS
home = expanduser("~")
bridge = CvBridge()
i = 0
size = None
status = -2

# Define paths
file_path = home + "/Pictures/" + "panda_data/"
rgb_path = file_path + 'panda_sim_vel/panda_rearranged_data/ycb_test/'
video_name = rgb_path + 'ycb_test_02.avi'  # Output video file path

frame_rate = 15  # Frames per second
extra_frames_for_last_image = 3  # Number of extra times to add the last image

if not os.path.exists(rgb_path):
    os.mkdir(rgb_path)

def statusCallback(msg):
    global status
    status = msg.data

def image_callback(img):
    global bridge, i, home, size, status

    fname = rgb_path + "image_" + str(i).zfill(6) + ".jpg"  # Zero padding for sorting
    cv_img = bridge.imgmsg_to_cv2(img, 'bgr8')
    cv2.imwrite(fname, cv_img)
    i += 1

def create_video_from_images():
    images = sorted(glob.glob(rgb_path + 'image_*.jpg'))  # Fetch images sorted by name
    print(images)
    if not images:
        print("No images found to create a video.")
        return

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
    print(f"Video saved as {video_name}")

def main(args):
    rospy.init_node('capture_images_for_video')

    # Subscriber for RGB images
    image_rgb_sub = rospy.Subscriber("/camera/color/image_raw", Image, image_callback, queue_size=1)
    rospy.on_shutdown(lambda: (time.sleep(2), create_video_from_images()))  # Ensure that all images are written before video creation

    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)