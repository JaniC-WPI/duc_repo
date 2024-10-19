#!/usr/bin/env python3

import rosbag
import matplotlib.pyplot as plt
import pandas as pd

# Load the rosbag file
bag = rosbag.Bag('control_data.bag')

# Initialize empty lists for storing data
goal_states = []
ds_data = []
joint_vel_data = []

# Extract relevant topics
for topic, msg, t in bag.read_messages(topics=['/current_goal_set_topic', '/ds_record', '/joint_vel']):
    if topic == '/current_goal_set_topic':
        goal_states.append(msg.data)
    elif topic == '/ds_record':
        ds_data.append(msg.data)
    elif topic == '/joint_vel':
        joint_vel_data.append(msg.data)

bag.close()

# Convert data to pandas DataFrame for easy plotting
df_goal_states = pd.DataFrame(goal_states, columns=['goal_state'])
df_ds = pd.DataFrame(ds_data, columns=['cp3_x', 'cp3_y', 'cp4_x', 'cp4_y', 'cp5_x', 'cp5_y', 'cp6_x', 'cp6_y'])
df_joint_vel = pd.DataFrame(joint_vel_data, columns=['Joint1', 'Joint2', 'Joint3'])

# Plotting the data
plt.figure(figsize=(10, 8))

# Plot for ds_record
plt.subplot(3, 1, 1)
plt.plot(df_ds.index, df_ds['cp3_x'], label='cp3_x')
plt.plot(df_ds.index, df_ds['cp3_y'], label='cp3_y')
plt.title('DS Record')
plt.legend()

# Plot for joint velocities
plt.subplot(3, 1, 2)
plt.plot(df_joint_vel.index, df_joint_vel['Joint1'], label='Joint 1')
plt.plot(df_joint_vel.index, df_joint_vel['Joint2'], label='Joint 2')
plt.plot(df_joint_vel.index, df_joint_vel['Joint3'], label='Joint 3')
plt.title('Joint Velocities')
plt.legend()

# Plot for goal states
plt.subplot(3, 1, 3)
plt.plot(df_goal_states.index, df_goal_states['goal_state'], label='Goal State', color='r')
plt.title('Goal States')
plt.legend()

plt.tight_layout()
plt.show()