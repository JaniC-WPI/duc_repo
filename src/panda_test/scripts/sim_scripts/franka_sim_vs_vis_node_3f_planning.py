#!/usr/bin/env python3

import roslib
import sys
import os
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Int32

bridge = CvBridge()

img_pub = rospy.Publisher("franka/vs/vis", Image, queue_size=1)

# Global variables for image capture and video creation
capture_images = False
images_captured = [] # all images saved
images_captured_status_2 = [] # only control images saved

status = -2
itr = 0
goal_image = None
current_goal_set = 0
cp1x, cp1y, cp2x, cp2y, cp3x, cp3y, cp4x, cp4y, cp5x, cp5y = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

no_of_actuators = rospy.get_param('vsbot/shape_control/no_of_actuators')
no_of_features = rospy.get_param('vsbot/shape_control/no_of_features')

# Current goal set callback to update the global current_goal_set variable
def goal_set_callback(goal_set_msg):
    global current_goal_set
    current_goal_set = goal_set_msg.data

def controlPointCallback(cp_msg):
    global cp1x, cp1y, cp2x, cp2y, cp3x, cp3y, cp4x, cp4y, cp5x, cp5y

    if (no_of_features ==2):
        cp1x, cp1y = cp_msg.data[0], cp_msg.data[1]
    elif (no_of_features == 4):
        cp1x, cp1y = cp_msg.data[0], cp_msg.data[1]
        cp2x, cp2y = cp_msg.data[2], cp_msg.data[3]
    elif (no_of_features == 6):
        cp1x, cp1y = cp_msg.data[0], cp_msg.data[1]
        cp2x, cp2y = cp_msg.data[2], cp_msg.data[3]
        cp3x, cp3y = cp_msg.data[4], cp_msg.data[5]
    elif (no_of_features == 8):
        cp1x, cp1y = cp_msg.data[0], cp_msg.data[1]
        cp2x, cp2y = cp_msg.data[2], cp_msg.data[3]
        cp3x, cp3y = cp_msg.data[4], cp_msg.data[5]
        cp4x, cp4y = cp_msg.data[6], cp_msg.data[7]
    elif (no_of_features == 10):
        cp1x, cp1y = cp_msg.data[0], cp_msg.data[1]
        cp2x, cp2y = cp_msg.data[2], cp_msg.data[3]
        cp3x, cp3y = cp_msg.data[4], cp_msg.data[5]
        cp4x, cp4y = cp_msg.data[6], cp_msg.data[7]
        cp5x, cp5y = cp_msg.data[8], cp_msg.data[9]

def goalImgCallback(img_msg):
    global goal_image
    goal_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")

def statusCallback(status_msg):
    global status, capture_images, images_captured, images_captured_status_2, current_goal_set
    status = status_msg.data
    if status == 1:
        capture_images = True
    elif status == 2:
        capture_images = True
    elif status == (2+current_goal_set):
        capture_images = True
        # images_captured_status_2 = [] # Reset the list for capturing images during status 2
    elif status == 50:
        capture_images = True
    elif status == -1 and capture_images:
        if images_captured:
            # Pass the path of the last captured image to the create_video_from_images function
            create_video_from_images(images_captured_status_2, "exp_vid.avi")
        else:
            print("No images available to create the video.")
            
        images_captured = []
        images_captured_status_2 = [] # Clear both lists after creating the video
        capture_images = False
        rospy.signal_shutdown("Shutting down")

def visCallback(msg):
    global bridge, img_pub, itr, capture_images, images_captured, goal_image, current_goal_set

    starter_img_path = "/home/jc-merlab/.ros/path_0.jpg"
    starter_img = cv2.imread(starter_img_path)
    cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    if status == 1:
        # text
        cv_img = cv2.putText(cv_img, 'Initialization', (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 1, cv2.LINE_AA)
        # cv_img = cv2.putText(cv_img, 'x2', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 2, cv2.LINE_AA)
        # cv_img = cv2.putText(cv_img, 'Frame #: '+str(itr), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 1, cv2.LINE_AA)
        # print("initialization!")

    # Apply goal image overlay if available and servoing
    if status == 2 and goal_image is not None:
        cv_img = cv2.addWeighted(cv_img, 1.0, starter_img, 1.4, 0)    
        cv_img = cv2.addWeighted(cv_img, 0.4, goal_image, 0.6, 0)    

    elif status == (2+current_goal_set) and goal_image is not None:
        cv_img = cv2.addWeighted(cv_img, 1.0, starter_img, 1.4, 0)    
        cv_img = cv2.addWeighted(cv_img, 0.4, goal_image, 0.6, 0)   

    elif status == 50:
        cv_img = cv2.addWeighted(cv_img, 0.4, goal_image, 0.6, 0)

    if (no_of_features==2):
        cv2.circle(cv_img, (int(cp1x), int(cp1y)), 5, (255,0,0), -1)
    elif (no_of_features==4):
        cv2.circle(cv_img, (int(cp1x), int(cp1y)), 5, (255,0,0), -1)
        cv2.circle(cv_img, (int(cp2x), int(cp2y)), 5, (0,255,0), -1)
    elif (no_of_features==6):
        cv2.circle(cv_img, (int(cp1x), int(cp1y)), 5, (255,0,0), -1)
        cv2.circle(cv_img, (int(cp2x), int(cp2y)), 5, (0,255,0), -1)
        cv2.circle(cv_img, (int(cp3x), int(cp3y)), 5, (255,255,0), -1)        
    if (no_of_features==8):
        cv2.circle(cv_img, (int(cp1x), int(cp1y)), 5, (255,0,0), -1)
        cv2.circle(cv_img, (int(cp2x), int(cp2y)), 5, (0,255,0), -1)
        cv2.circle(cv_img, (int(cp3x), int(cp3y)), 5, (255,255,0), -1)
        cv2.circle(cv_img, (int(cp4x), int(cp4y)), 5, (0,255,255), -1)
    elif (no_of_features==10):
        cv2.circle(cv_img, (int(cp1x), int(cp1y)), 5, (255,0,0), -1)
        cv2.circle(cv_img, (int(cp2x), int(cp2y)), 5, (0,255,0), -1)
        cv2.circle(cv_img, (int(cp3x), int(cp3y)), 5, (255,255,0), -1)
        cv2.circle(cv_img, (int(cp4x), int(cp4y)), 5, (0,255,255), -1)
        cv2.circle(cv_img, (int(cp5x), int(cp5y)), 5, (255,0,255), -1)

    if capture_images:
        fname = "frame_{:04d}.png".format(itr)
        cv_img = cv2.putText(cv_img, str(current_goal_set+1), (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2, cv2.LINE_AA)
        cv2.imwrite(fname, cv_img)
        images_captured.append(fname)
        if (status == 2) or status == (2+current_goal_set) or (status == 50): # Add images to the status 2 list only if currently in status 2
            images_captured_status_2.append(fname)
            
        itr += 1
    # print(images_capturedand_status_2)
    try:
        ros_img = bridge.cv2_to_imgmsg(cv_img, "bgr8")
    except CvBridgeError as e:
        print(e)

    img_pub.publish(ros_img)

def create_video_from_images(image_files, output_file, initial_frame_repeats=60):
    """
    Creates a video from a list of image files with initial frames being a blended image of:
    sim_published_goal_image.jpg, last captured image, and init_img_path.

    :param init_img_path: Path to the last captured image from images_captured list.
    :param image_files: List of image filenames to include in the video.
    :param output_file: Name of the output video file.
    :param initial_frame_repeats: Number of times the initial blended frame should be repeated.
    """
    if not image_files:
        print("No images to process for video creation.")
        return
    
    # Read images using the paths provided
    img_1 = cv2.imread("/home/jc-merlab/.ros/sim_published_goal_image.jpg")  # Initial published goal image
    img_2 = cv2.imread(image_files[0])  # Last image from images_captured
    img_3 = cv2.imread(image_files[-1])  # Last image from images_captured_status_2

    # Check if the images are read correctly
    if img_1 is None:
        print(f"Error: Could not read sim_published_goal_image.jpg.")
        return

    if img_2 is None:
        print(f"Error: Could not read the initial image from path: {init_img_path}")
        return

    if img_3 is None:
        print(f"Error: Could not read the last image from image files list.")
        return

    # Ensure all images have the same dimensions before blending
    if img_1.shape != img_2.shape:
        print(f"Resizing img_2 to match dimensions: {img_2.shape} -> {img_1.shape}")
        img_2 = cv2.resize(img_2, (img_1.shape[1], img_1.shape[0]))

    if img_1.shape != img_3.shape:
        print(f"Resizing img_3 to match dimensions: {img_3.shape} -> {img_1.shape}")
        img_3 = cv2.resize(img_3, (img_1.shape[1], img_1.shape[0]))

    # Blend the images sequentially: img_1 + img_2, then the result + img_3
    initial_img_1 = cv2.addWeighted(img_1, 0.6, img_2, 0.8, 0)  # Blend sim_published_goal_image with init_img_path
    blended_initial_img = cv2.addWeighted(initial_img_1, 0.6, img_3, 0.8, 0)  # Blend the result with the last image

    # Get the video size from the final blended image
    height, width, _ = blended_initial_img.shape
    size = (width, height)

    # Create the video writer object
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), 20, size)

    # Write the blended initial image `initial_frame_repeats` times to the video
    for _ in range(initial_frame_repeats):
        out.write(blended_initial_img)

    # Add all subsequent images to the video
    for filename in image_files:
        img = cv2.imread(filename)
        if img is not None:
            out.write(img)
        else:
            print(f"Warning: Skipped {filename} as it could not be read.")

    # Release the video writer object and save the video
    out.release()
    print(f"Video saved as {output_file} with {initial_frame_repeats} initial blended frames.")

def main(args):
    rospy.init_node('visualizer', anonymous=True)
    print("Initialized vis")

    img_sub = rospy.Subscriber("camera/color/image_raw", Image, visCallback, queue_size=1)
    status_sub = rospy.Subscriber("vsbot/status", Int32, statusCallback, queue_size=1)
    goal_set_sub = rospy.Subscriber("current_goal_set_topic", Int32, goal_set_callback, queue_size=1)
    goal_sub = rospy.Subscriber("franka/sim_goal_image", Image, goalImgCallback, queue_size=1)
    cp_sub = rospy.Subscriber("vsbot/control_points", Float64MultiArray, controlPointCallback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == "__main__":
    main(sys.argv)
