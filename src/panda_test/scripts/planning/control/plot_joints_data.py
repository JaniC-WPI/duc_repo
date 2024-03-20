import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_joint_config_from_csv(file_path):
    """
    Reads joint configurations from a CSV file.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        np.array: Array of joint configurations.
    """
    return np.loadtxt(file_path, delimiter=',', skiprows=0)  # No header row, tab-delimited

def read_all_joints_from_csv(file_path):
    """
    Reads joint configurations from a CSV file.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        np.array: Array of joint configurations.
    """
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    data = data[:, 1:]
    return data


def plot_joint_configs(g1_joint_configs, g2_joint_configs, all_joints):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the first path
    ax.plot(g1_joint_configs[:, 0], g1_joint_configs[:, 1], g1_joint_configs[:, 2], 'b-o', label='Euclidean')

    # Plotting the second path
    ax.plot(g2_joint_configs[:, 0], g2_joint_configs[:, 1], g2_joint_configs[:, 2], 'g-o', label='Custom')

    ax.plot(all_joints[:, 0], all_joints[:, 1], all_joints[:, 2], 'r-o', label='Control Motion')

    # Highlight the start and goal for the first path
    ax.scatter(g1_joint_configs[0, 0], g1_joint_configs[0, 1], g1_joint_configs[0, 2], color='magenta', s=50, label='Start')
    ax.scatter(g1_joint_configs[-1, 0], g1_joint_configs[-1, 1], g1_joint_configs[-1, 2], color='gold', s=50, label='Goal')

    # Optionally, highlight the start and goal for the second path if they are different
    # ax.scatter(g2_joint_configs[0, 0], g2_joint_configs[0, 1], g2_joint_configs[0, 2], color='magenta', s=50)
    # ax.scatter(g2_joint_configs[-1, 0], g2_joint_configs[-1, 1], g2_joint_configs[-1, 2], color='gold', s=50)

    ax.set_xlabel('Joint 1')
    ax.set_ylabel('Joint 2')
    ax.set_zlabel('Joint 3')
    ax.legend()
    plt.show()


# Assuming you have the path to your CSV files
g1_joint_configs_csv_path = '/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps/euc_3/g1_joint_configs.csv'
g2_joint_configs_csv_path = '/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps/cust_3/g2_joint_configs.csv'

all_joints_path = '/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps/cust_3/3/joint_angles.csv'

# Read joint configurations from CSV files
g1_joint_configs = read_joint_config_from_csv(g1_joint_configs_csv_path)
g2_joint_configs = read_joint_config_from_csv(g2_joint_configs_csv_path)
all_joints = read_all_joints_from_csv(all_joints_path)

# Call the plot function
plot_joint_configs(g1_joint_configs, g2_joint_configs, all_joints)
