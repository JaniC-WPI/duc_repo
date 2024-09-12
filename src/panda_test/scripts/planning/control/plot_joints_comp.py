import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_joint_configurations(ax, file_path, label, color, start_marker='o', goal_marker='X'):
    """
    Reads a CSV file containing joint configurations and plots them in 3D.

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
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Extract joint angles
    joint1 = df['Joint 1']
    joint2 = df['Joint 2']
    joint3 = df['Joint 3']
    distances = df['Distance to next Joint Angles']

    # Plot joint configurations in 3D
    ax.plot(joint1, joint2, joint3, label=label, color=color)
    ax.scatter(joint1, joint2, joint3, color=color)

    # Mark the start and goal joint angles
    if label == 'Ground Truth':
        ax.scatter(joint1.iloc[0], joint2.iloc[0], joint3.iloc[0], color='green', marker=start_marker, s=100, label= 'Start')
        ax.scatter(joint1.iloc[-1], joint2.iloc[-1], joint3.iloc[-1], color='purple', marker=goal_marker, s=100, label= 'Goal')


    # Add distance labels between joints
    # for i in range(len(joint1) - 1):
    #     # Calculate the midpoint between two consecutive joints
    #     mid_x = (joint1.iloc[i] + joint1.iloc[i+1]) / 2
    #     mid_y = (joint2.iloc[i] + joint2.iloc[i+1]) / 2
    #     mid_z = (joint3.iloc[i] + joint3.iloc[i+1]) / 2

    #     # Add text label for the distance to next joint angle
    #     ax.text(mid_x, mid_y, mid_z, f'{distances.iloc[i]:.2f}', color='black', fontsize=8)

def main():
    # Paths to the CSV files
    # file_path1 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/with_obs/6/save_distances.csv'  
    # file_path2 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/with_obs/6/save_distances.csv'
    # file_path3 =  '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/with_obs/6/save_distances.csv'

    file_path1 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/custom/with_obs/302/save_distances.csv'
    file_path2 = '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/euclidean/with_obs/302/save_distances.csv'
    file_path3 =  '/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/ground_truth/with_obs/302/save_distances.csv'


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
    ax.set_title('3D Joint Configurations Comparison')
    ax.legend(loc='upper right')

    # Save the plot
    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/configurations_and_goals/joint_plots/with_obs/jt_dist_6.png', dpi=300)  # Replace with your desired save path and format

    # Show the plot
    plt.show()

# Run the main function
if __name__ == "__main__":
    main()