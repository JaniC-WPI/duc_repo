#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Global variables to store status and current_goal_set
status = -2
current_goal_set = 0
cv_goal_img = None
last_status = None

# Status callback to update the global status variable
def status_callback(status_msg):
    global status
    # print("Current Status", status)
    status = status_msg.data

# Current goal set callback to update the global current_goal_set variable
def goal_set_callback(goal_set_msg):
    global current_goal_set
    current_goal_set = goal_set_msg.data

def main():
    global cv_goal_img, last_status
    # Initialize the node
    rospy.init_node('sim_goal_image_publish_node')
    bridge = CvBridge()

    # Initialize subscribers for status and current_goal_set
    rospy.Subscriber("vsbot/status", Int32, status_callback, queue_size=1)
    rospy.Subscriber("current_goal_set_topic", Int32, goal_set_callback, queue_size=1)

    # Declaring the publisher for image
    goal_image_pub = rospy.Publisher('franka/sim_goal_image', Image, queue_size=1)

    # Assigning a rate at which the image will be published
    r = rospy.get_param('vsbot/estimation/rate')
    rate = rospy.Rate(r)
    # Load the final goal image once at the start
    final_goal_image_path = "/home/jc-merlab/.ros/sim_published_goal_image.jpg"
    final_goal_img = cv2.imread(final_goal_image_path)

    if final_goal_img is None:
        rospy.logerr(f"Failed to load final goal image from {final_goal_image_path}")
    else:
        rospy.loginfo("Final goal image loaded successfully.")

    while not rospy.is_shutdown():
        # print("which image is it:", cv_goal_img)
        # Check the status and set the goal image accordingly
        if status == 50 and final_goal_img is not None:
            # Load the final goal image once if it hasn't been loaded already
            # Use the final goal image for continuous publishing
            cv_goal_img = final_goal_img
            rospy.loginfo(f"Publishing final goal image continuously, Status: {status}")
        elif status == (2 + current_goal_set):
            # For intermediate goals, load sim_intermediate_goal_image_i.jpg where i = current_goal_set + 1
            image_path = f"/home/jc-merlab/.ros/sim_intermediate_goal_image_{current_goal_set + 1}.jpg"
            cv_goal_img = cv2.imread(image_path)
            if cv_goal_img is None:
                rospy.logerr(f"Failed to load intermediate goal image from {image_path}")
                continue
            rospy.loginfo(f"Publishing intermediate goal image, Status: {status}")
        elif status == 2:
            # For the first goal, set to sim_intermediate_goal_image_1.jpg
            image_path = f"/home/jc-merlab/.ros/sim_intermediate_goal_image_1.jpg"
            cv_goal_img = cv2.imread(image_path)
            print("is cv_goal_image None: ", cv_goal_img)
            if cv_goal_img is None:
                rospy.logerr(f"Failed to load first goal image from {image_path}")
                continue
            rospy.loginfo(f"Publishing first goal image, Status: {status}")
        else:
            # If the status is not one of the expected values, skip publishing
            rospy.loginfo(f"Skipping image publication, Status: {status}")
            rate.sleep()
            continue
        # Check if cv_goal_img is loaded successfully
        if cv_goal_img is not None:
            # Convert the image to a ROS message
            try:
                published_image = bridge.cv2_to_imgmsg(cv_goal_img, "bgr8")
            except CvBridgeError as e:
                rospy.logerr(f"Failed to convert image: {e}")
                continue
                
            goal_image_pub.publish(published_image)
        
        rate.sleep()

if __name__ == '__main__':
    main()