import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_joint_configurations(ax, file_path, label, color, start_marker='o', goal_marker='X'):
    """
    Reads a CSV file containing joint configurations, skips the first 50 iterations,
    and plots them in 3D.

    Args:
    - ax (Axes3D): The 3D axes object to plot on.
    - file_path (str): Path to the CSV file.
    - label (str): Label for the plot legend.
    - color (str): Color of the plot line.
    - start_marker (str): Marker style for the start joint.
    - goal_marker (str): Marker style for the goal joint.

    Returns:
    - None
    """
    # Read the CSV file, skip first 50 rows and extract 2nd, 3rd, 4th columns as Joint1, Joint2, Joint3
    df = pd.read_csv(file_path, header=None, usecols=[1, 2, 3], skiprows=50, names=['Joint 1', 'Joint 2', 'Joint 3'])

    # Extract joint angles
    joint1 = df['Joint 1']
    joint2 = df['Joint 2']
    joint3 = df['Joint 3']

    # Plot joint configurations in 3D
    ax.plot(joint1, joint2, joint3, label=label, color=color)
    ax.scatter(joint1, joint2, joint3, color=color)

    ax.scatter(joint1.iloc[0], joint2.iloc[0], joint3.iloc[0], color='green', marker=start_marker, s=100)
    ax.scatter(joint1.iloc[-1], joint2.iloc[-1], joint3.iloc[-1], color='purple', marker=goal_marker, s=100)


    # Mark the start and goal joint angles
    if label == 'Ground Truth':
        ax.scatter(joint1.iloc[0], joint2.iloc[0], joint3.iloc[0], color='green', marker=start_marker, s=100, label= 'Start')
        ax.scatter(joint1.iloc[-1], joint2.iloc[-1], joint3.iloc[-1], color='purple', marker=goal_marker, s=100, label= 'Goal')

def main():
    # Path to your data file
    file_path1 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/10/joint_angles.csv' 
    file_path2 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps_09_08_2024/10/joint_angles.csv' 
    file_path3 =  '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/4/joint_angles.csv'  


    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot joint configurations from the first CSV file
    plot_joint_configurations(ax, file_path1, label='Custom', color='blue')

    # Plot joint configurations from the second CSV file
    plot_joint_configurations(ax, file_path2, label='Euclidean', color='red')

    # Plot joint configurations from the third CSV file
    plot_joint_configurations(ax, file_path3, label='Ground Truth', color='green')

    # Set labels
    ax.set_xlabel('Joint 1')
    ax.set_ylabel('Joint 2')
    ax.set_zlabel('Joint 3')
    ax.set_title('3D Joint Configurations')
    ax.legend(loc='upper right')

    # Show the plot
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()