#!/usr/bin/env python3

import json
import numpy as np
import cv2
import heapq
import os
import networkx as nx
import time
from sklearn.neighbors import KDTree
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()
# Global publisher
image_pub = None

# Detect a red ball in an image
def detect_red_ball(image):
    # image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Red color might be in two ranges
    lower_red_1 = np.array([0, 50, 50])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 50, 50])
    upper_red_2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = mask1 + mask2  # Combine masks
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:        
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)  # Green circle with thickness of 2
        # Draw the center of the circle
        center_radius = 5  # Radius of the center point
        cv2.circle(image, (int(x), int(y)), center_radius, (255, 0, 0), -1)  # Blue filled circle for the center

        # Convert the modified image back to a ROS message and publish it
        modified_img_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
        image_pub.publish(modified_img_msg)

        print(int(x), int(y), int(radius))

        return (int(x), int(y), int(radius))
        
    return None


def image_callback(ros_img):

        cv_img = bridge.imgmsg_to_cv2(ros_img, "bgr8")
        obstacle_info = detect_red_ball(cv_img)
        if obstacle_info is not None:
            obstacle_center, obstacle_radius = obstacle_info[:2], obstacle_info[2]
            print(obstacle_center)
        else:
            print("No red ball detected in the image.")
            obstacle_center, obstacle_radius = None, None


def main():
    global image_pub
    # Initialize the node
    rospy.init_node('kp_inference_gen')
    print("is main getting called")

    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, image_callback, queue_size=1)
    # Initialize the publisher
    image_pub = rospy.Publisher("/processed_image", Image, queue_size=1)


    rospy.spin()

if __name__=='__main__':
    main()
