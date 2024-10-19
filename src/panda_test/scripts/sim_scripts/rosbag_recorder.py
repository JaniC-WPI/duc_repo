#!/usr/bin/env python3.8
import rospy
import rosbag
import csv
from std_msgs.msg import Float32, Float64MultiArray, Int32
import os
from os.path import expanduser

# Global variables to store rosbag objects and file paths
bags = {}
csv_writers = {}
csv_files = {}
control_rate = None

def callback(msg, topic_name):
    """Callback function to write messages to the appropriate rosbag and CSV file."""
    if topic_name in bags:
        # Write to the rosbag
        bags[topic_name].write(topic_name, msg)

        # Convert the message to a CSV-friendly format and write to the CSV file
        if topic_name in csv_writers:
            if isinstance(msg, Float64MultiArray) or isinstance(msg, Float32):
                data_row = [msg.data] if isinstance(msg, Float32) else msg.data
                csv_writers[topic_name].writerow(data_row)
            elif isinstance(msg, Int32):
                csv_writers[topic_name].writerow([msg.data])

def setup_rosbags_and_csvs(output_dir):
    """Set up rosbag files and CSV files for each topic."""
    global bags, csv_writers, csv_files

    # Create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create rosbag files for each topic and set up CSV files
    topics = [
        "joint_vel", "ds_record", "dr_record", "J_modelerror",
        "servoing_error", "vsbot_status", "vsbot_control_points",
        "current_goal_set_topic"
    ]

    for topic in topics:
        # Set up rosbags
        bags[topic] = rosbag.Bag(os.path.join(output_dir, f"{topic}.bag"), 'w')

        # Set up CSV files
        csv_file_path = os.path.join(output_dir, f"{topic}.csv")
        csv_files[topic] = open(csv_file_path, 'w', newline='')  # Open CSV file
        csv_writers[topic] = csv.writer(csv_files[topic])

        # Write CSV header based on topic type (simple headers here for demonstration)
        if topic == "J_modelerror" or topic == "vsbot_status" or topic == "current_goal_set_topic":
            csv_writers[topic].writerow(["data"])  # Single value
        else:
            csv_writers[topic].writerow(["data_{}".format(i) for i in range(10)])  # Assuming max 10 elements, adjust as needed

def close_rosbags_and_csvs():
    """Close all rosbag and CSV files."""
    for bag in bags.values():
        bag.close()
    for file in csv_files.values():
        file.close()

def main():
    global control_rate

    # Initialize ROS node
    rospy.init_node('rosbag_recorder_to_csv', anonymous=True)

    # Get the control rate and output directory from parameters
    control_rate = rospy.get_param("vsbot/estimation/rate")
    dir = '~/.ros'
    output_dir = os.path.expanduser(dir)

    # Setup rosbag files
    setup_rosbags_and_csvs(output_dir)

    # Subscribers for each topic
    rospy.Subscriber("joint_vel", Float64MultiArray, callback, "joint_vel")
    rospy.Subscriber("ds_record", Float64MultiArray, callback, "ds_record")
    rospy.Subscriber("dr_record", Float64MultiArray, callback, "dr_record")
    rospy.Subscriber("J_modelerror", Float32, callback, "J_modelerror")
    rospy.Subscriber("servoing_error", Float64MultiArray, callback, "servoing_error")
    rospy.Subscriber("vsbot/status", Int32, callback, "vsbot_status")
    rospy.Subscriber("vsbot/control_points", Float64MultiArray, callback, "vsbot_control_points")
    rospy.Subscriber("current_goal_set_topic", Int32, callback, "current_goal_set_topic")

    # Control rate
    rate = rospy.Rate(control_rate)

    # Keep the node running
    try:
        while not rospy.is_shutdown():
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Ensure rosbag files are properly closed on shutdown
        close_rosbags_and_csvs()

    rospy.spin()

if __name__ == '__main__':
    main()