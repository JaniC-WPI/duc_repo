#!/usr/bin/env python3

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest

def publish_joint_trajectory():
    # Initialize the ROS node
    rospy.init_node('joint_trajectory_publisher', anonymous=True)

    # Create a publisher for the /position_joint_trajectory_controller/command topic
    pub = rospy.Publisher('/position_joint_trajectory_controller/command', JointTrajectory, queue_size=10)

    # Set the rate at which to publish messages (e.g., once every 5 seconds)
    rate = rospy.Rate(0.2)  # 0.2 Hz (5 seconds interval)

    while not rospy.is_shutdown():
        # Create a JointTrajectory message
        trajectory_msg = JointTrajectory()
        trajectory_msg.header.stamp = rospy.Time.now()

        # Define joint names according to your robot's configuration
        trajectory_msg.joint_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 
            'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
        ]

        # Create a single JointTrajectoryPoint
        point = JointTrajectoryPoint()
        point.positions = [0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.2]  # Desired joint positions
        point.velocities = [0.0] * 7  # Assuming zero velocities for each joint
        point.accelerations = [0.0] * 7  # Assuming zero accelerations for each joint
        point.effort = [0.0] * 7  # Assuming zero effort for each joint
        point.time_from_start = rospy.Duration(10)  # 5 seconds to reach the target positions

        # Add the point to the trajectory message
        trajectory_msg.points.append(point)

        # Publish the trajectory message
        pub.publish(trajectory_msg)
        rospy.loginfo("Published JointTrajectory to /position_joint_trajectory_controller/command")

        # Sleep for the specified duration
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_joint_trajectory()
    except rospy.ROSInterruptException:
        pass