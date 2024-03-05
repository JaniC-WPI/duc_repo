import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import numpy as np
import os
import networkx as nx
from sklearn.neighbors import KDTree, BallTree


def load_matched_configurations(directory):
    # Initialize empty lists for configurations
    kp_configurations = []
    jt_configurations = []

    # Temporary dictionary to hold joint angles keyed by identifier
    temp_jt_configurations = {}

    # First pass: Load joint angles into temporary dictionary
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('_joint_angles.json'):
            identifier = filename.replace('_joint_angles.json', '')
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                temp_jt_configurations[identifier] = np.array(data['joint_angles'])

    # Second pass: Match and load keypoints configurations
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.json') and not filename.endswith('_joint_angles.json') and not filename.endswith('_vel.json'):
            identifier = filename.replace('.json', '')
            if identifier in temp_jt_configurations:
                with open(os.path.join(directory, filename), 'r') as file:
                    data = json.load(file)
                    keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]
                    kp_configurations.append(np.array(keypoints))
                    jt_configurations.append(temp_jt_configurations[identifier])

    print(kp_configurations[0:10], jt_configurations[0:10])
    return kp_configurations, jt_configurations

def plot_combined(keypoints, joint_angles, title='Combined Visualization of Keypoints and Joint Angles'):
    fig = plt.figure(figsize=(14, 6))

    # Plotting 2D keypoints
    ax1 = fig.add_subplot(121)
    ax1.scatter(keypoints[:, 0], keypoints[:, 1], c='blue', label='2D Keypoints')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('2D Keypoints')
    ax1.axis('equal')
    
    # Plotting 3D joint angles
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(joint_angles[0], joint_angles[1], joint_angles[2], c='red', label='3D Joint Angles')
    ax2.set_xlabel('Joint 1')
    ax2.set_ylabel('Joint 2')
    ax2.set_zlabel('Joint 3')
    ax2.set_title('3D Joint Angles')

    plt.suptitle(title)
    plt.show()

if __name__ == "__main__":
    # Load configurations from JSON files
    directory = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_planning_kprcnn/' 
    kp_configurations, joint_angles = load_matched_configurations(directory)
    print(np.array(kp_configurations).shape)
    print(np.array(joint_angles).shape)
    plot_combined(kp_configurations, joint_angles)