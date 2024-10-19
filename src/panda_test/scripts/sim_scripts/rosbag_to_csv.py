#!/usr/bin/env python3
import rosbag
import csv
import os
import rospy
from std_msgs.msg import Float32, Float64MultiArray, Int32

def convert_rosbag_to_csv(rosbag_dir, csv_output_dir):
    """Convert rosbags in a directory to individual CSV files."""
    # Create output directory if it doesn't exist
    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)

    # List all .bag files in the specified directory
    for bag_file in os.listdir(rosbag_dir):
        if bag_file.endswith(".bag"):
            bag_path = os.path.join(rosbag_dir, bag_file)
            topic_name = os.path.splitext(bag_file)[0]  # Use the filename as the topic name

            # Skip the current_goal_set topic since it will be merged
            if topic_name == "current_goal_set_topic":
                continue

            csv_file_path = os.path.join(csv_output_dir, f"{topic_name}.csv")

            # Open the rosbag and create the CSV file
            with rosbag.Bag(bag_path, 'r') as bag, open(csv_file_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)

                # Write headers for CSV
                csv_writer.writerow(["current_goal", "timestamp", "data"])  # Adjust headers if necessary

                # Open the current_goal_set bag to merge data
                current_goal_bag_path = os.path.join(rosbag_dir, "current_goal_set_topic.bag")
                current_goal_data = []

                # Read current_goal_set data
                with rosbag.Bag(current_goal_bag_path, 'r') as goal_bag:
                    for topic, msg, t in goal_bag.read_messages(topics=["current_goal_set_topic"]):
                        current_goal_data.append((t.to_sec(), msg.data))

                # Iterate through the messages in the rosbag
                for topic, msg, t in bag.read_messages(topics=[topic_name]):
                    # Find the closest timestamp in current_goal_data
                    closest_goal = min(current_goal_data, key=lambda x: abs(x[0] - t.to_sec()))[1]

                    # Extract message data and write to CSV
                    if isinstance(msg, Float64MultiArray):
                        data_row = [closest_goal, t.to_sec()] + list(msg.data)  # Current goal, time, and data elements
                        csv_writer.writerow(data_row)
                    elif isinstance(msg, Float32):
                        csv_writer.writerow([closest_goal, t.to_sec(), msg.data])  # Current goal, time, and single data value
                    elif isinstance(msg, Int32):
                        csv_writer.writerow([closest_goal, t.to_sec(), msg.data])  # Current goal, time, and single data value

def main():
    rospy.init_node('rosbag_to_csv', anonymous=True)

    # Get the rosbag directory and csv output directory from ROS parameters
    rosbag_dir = "/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/1/"  # Directory containing rosbags
    csv_output_dir = "/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/1/"  # Directory to store CSVs

    # Convert the rosbags to CSV files
    convert_rosbag_to_csv(rosbag_dir, csv_output_dir)

    rospy.loginfo(f"Conversion completed. CSV files are saved in: {csv_output_dir}")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass