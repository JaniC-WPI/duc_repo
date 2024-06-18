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

cp1x, cp1y, cp2x, cp2y, cp3x, cp3y, cp4x, cp4y, cp5x, cp5y = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

no_of_actuators = rospy.get_param('vsbot/shape_control/no_of_actuators')
no_of_features = rospy.get_param('vsbot/shape_control/no_of_features')

def controlPointCallback(cp_msg):
    global cp1x, cp1y, cp2x, cp2y, cp3x, cp3y, cp4x, cp4y, cp5x, cp5y

    cp1x, cp1y = cp_msg.data[0], cp_msg.data[1]
    cp2x, cp2y = cp_msg.data[2], cp_msg.data[3]
    cp3x, cp3y = cp_msg.data[4], cp_msg.data[5]

    if (no_of_features == 8):
        cp4x, cp4y = cp_msg.data[6], cp_msg.data[7]
    elif (no_of_features == 10):
        cp4x, cp4y = cp_msg.data[6], cp_msg.data[7]
        cp5x, cp5y = cp_msg.data[8], cp_msg.data[9]

def goalImgCallback(img_msg):
    global goal_image
    goal_image = bridge.imgmsg_to_cv2(img_msg, "bgr8")

def statusCallback(status_msg):
    global status, capture_images, images_captured, images_captured_status_2
    status = status_msg.data
    if status == 1:
        capture_images = True
    elif status == 2:
        capture_images = True
        # images_captured_status_2 = [] # Reset the list for capturing images during status 2
    elif status == -1 and capture_images:
        create_video_from_images(images_captured_status_2, "exp_vid.avi")
        # if images_captured_status_2: # Ensure we have images from status 2 before creating a video
            
        images_captured = []
        images_captured_status_2 = [] # Clear both lists after creating the video
        capture_images = False
        rospy.signal_shutdown("Shutting down")

def visCallback(msg):
    global bridge, img_pub, itr, capture_images, images_captured, goal_image

    cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    if status == 1:
        # text
        cv_img = cv2.putText(cv_img, 'Initialization', (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 1, cv2.LINE_AA)
        # cv_img = cv2.putText(cv_img, 'x2', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 2, cv2.LINE_AA)
        # cv_img = cv2.putText(cv_img, 'Frame #: '+str(itr), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 1, cv2.LINE_AA)
        # print("initialization!")

    # Apply goal image overlay if available and servoing
    if status == 2 and goal_image is not None:
        cv_img = cv2.addWeighted(cv_img, 0.4, goal_image, 0.6, 0)

    cv2.circle(cv_img, (int(cp1x), int(cp1y)), 5, (255,0,0), -1)
    cv2.circle(cv_img, (int(cp2x), int(cp2y)), 5, (0,255,0), -1)
    cv2.circle(cv_img, (int(cp3x), int(cp3y)), 5, (255,255,0), -1)
    if (no_of_features==8):
        cv2.circle(cv_img, (int(cp4x), int(cp4y)), 5, (0,255,255), -1)
    elif (no_of_features==10):
        cv2.circle(cv_img, (int(cp4x), int(cp4y)), 5, (0,255,255), -1)
        cv2.circle(cv_img, (int(cp5x), int(cp5y)), 5, (255,0,255), -1)

    if capture_images:
        fname = "frame_{:04d}.png".format(itr)
        cv2.imwrite(fname, cv_img)
        images_captured.append(fname)
        if status == 2: # Add images to the status 2 list only if currently in status 2
            images_captured_status_2.append(fname)
            
        itr += 1
    # print(images_captured_status_2)
    try:
        ros_img = bridge.cv2_to_imgmsg(cv_img, "bgr8")
    except CvBridgeError as e:
        print(e)

    img_pub.publish(ros_img)

def create_video_from_images(image_files, output_file):
    if not image_files:
        return

    img = cv2.imread(image_files[0])
    height, width, layers = img.shape
    size = (width, height)

    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), 20, size)

    for filename in image_files:
        img = cv2.imread(filename)
        out.write(img)
    out.release()
    print("Video saved as", output_file)

    # for filename in image_files:
    #     os.remove(filename)
    # print("Temporary image files removed.")

def main(args):
    rospy.init_node('visualizer', anonymous=True)
    print("Initialized vis")

    img_sub = rospy.Subscriber("camera/color/image_raw", Image, visCallback, queue_size=1)
    status_sub = rospy.Subscriber("vsbot/status", Int32, statusCallback, queue_size=1)
    goal_sub = rospy.Subscriber("franka/sim_goal_image", Image, goalImgCallback, queue_size=1)
    cp_sub = rospy.Subscriber("vsbot/control_points", Float64MultiArray, controlPointCallback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == "__main__":
    main(sys.argv)
