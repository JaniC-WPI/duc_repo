#!/usr/bin/env python3

import roslib
import sys
# import rospy
import numpy as np
import matplotlib.pyplot as plt
import csv
from std_msgs.msg import Float64MultiArray, Int64
from PIL import Image
import math


def main(args):
    
    # Initialize ROS
    # rospy.init_node('franka_plotter')

    with open('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/qhat.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader, None)  # Skip the first row (header)
        jacobian_data = [list(map(float, row)) for row in csv_reader]

    # Convert to a NumPy array for easier slicing
    jacobian_data = np.array(jacobian_data)

    # Number of features and joints
    num_features = 8
    num_joints = 3

    # Reshape the data to have the shape (number of iterations, number of features, number of joints)
    num_iterations = len(jacobian_data) // num_features
    jacobian_data_reshaped = jacobian_data[:num_iterations * num_features].reshape((num_iterations, num_features, num_joints))
    assert len(jacobian_data) % num_features == 0, "Incomplete Data set for features"
    print(jacobian_data_reshaped.shape)
    # Plot data for each joint
    for j in range(num_joints):
        plt.figure(figsize=(10, 6))
        for i in range(num_features):
            print(jacobian_data_reshaped[:, i, j])
            plt.plot(jacobian_data_reshaped[:, i, j], label=f'Feature {i+1}')
        plt.title(f'Joint {j+1} Influence over Time')
        plt.xlabel('Iterations')
        plt.ylabel('Influence value')
        plt.grid()
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)
        plt.tight_layout()
        plt.savefig(f'/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/joint_{j+1}_influence_on_features.png', dpi=300)

    with open('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/qhat_feat.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader, None)  # Skip the first row (header)
        pseduo_jacobian_data = [list(map(float, row)) for row in csv_reader]

    pseudo_jacobian_data = np.array(pseduo_jacobian_data).T
    pseudo_jacobian_data = pseudo_jacobian_data.reshape((-1, 3, 8))

    # Number of iterations is the number of times a full 3x8 jacobian matrix is recorded.
    num_iterations = pseudo_jacobian_data.shape[0]
    print(num_iterations)
    num_pairs = num_features // 2

    # # Plotting the influence of each joint on each feature over time
    # for joint in range(3):
    #     plt.figure(figsize=(14, 7))
    #     for feature in range(8):
    #         plt.plot(jacobian_data[:, joint, feature], label=f'Feature {feature+1}')

    #     plt.title(f'Joint {joint+1} Influence over Time')
    #     plt.xlabel('Iterations')
    #     plt.ylabel(f'Joint {joint+1} Influence Value')
    #     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=4)
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f'/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/feature_influence_on_joint_{joint+1}.png') # Uncomment to save the figures.    

    # Plotting the influence of pairs of features on each joint over time
    features_per_plot = 2  # Assuming num_features is always even and = 8 in your case.

    for joint in range(num_joints):
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex=True)
        fig.suptitle(f'Joint {joint+1} Influence over Time')

        for pair_idx in range(0, num_features, features_per_plot):
            row = (pair_idx // features_per_plot) // 2
            col = (pair_idx // features_per_plot) % 2
            ax = axs[row, col]

            feature1 = pair_idx
            feature2 = pair_idx + 1

            ax.plot(pseudo_jacobian_data[:, joint, feature1], label=f'Feature {feature1+1}')
            ax.plot(pseudo_jacobian_data[:, joint, feature2], label=f'Feature {feature2+1}')
            ax.set_title(f'Features {feature1+1} & {feature2+1}')
            ax.legend()
            ax.grid(True)

        # Set common labels
        for ax in axs.flat:
            ax.set(xlabel='Iterations', ylabel='Influence Value')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/joint_{joint+1}_feature_pairs_influence.png')  # Save the figure

     # Reshape data to 8x3 format for each set of jacobian matrices    

    # plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/qhat.png', dpi=300)


    err_data = []
    # read feature error
    # with open('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/8/err.csv','r') as csvfile:
    with open('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/err.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        for row in plots:
            err_data.append(row)

    # Post process the list
    err_data = err_data[1:]
    err_data = [list( map(float,i) ) for i in err_data]
    err_data = np.array(err_data)
    
    # compute error norm
    err_norm = []
    for row in range(len(err_data)):
        sum = 0
        for err in err_data[row]:
            sum += float(err)**2
        err_norm.append(math.sqrt(sum))

    # Identify indices where 'current_goal_set' changes
    changes_err = np.where(np.diff(err_data[:, 0]) != 0)[0] + 1   

    # Extract 'current_goal_sets' and 'err_data/model_error' without the 'current_goal_set' column
    current_goal_sets_err = err_data[:, 0]
    err_data = err_data[:, 1:]
    # current_goal_sets_model = model_error[:, 0]
    # model_error = model_error[:, 1:]

    # create plot axes
    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax1 = axes[0]
    ax2 = axes[1]
    fig.tight_layout(pad=4.0)

    

    # Plot error norm   
    ax1.plot(err_norm, 'b',linewidth=1.6)
    ax1.margins(x = 0.0, y=0.0)
    ax1.set_ylabel('Error norm', fontsize=13)
    ax1.set_xlabel('Control loop iteration #', fontsize=13)
    ax1.grid()    
    ax1.set_yticks(np.arange(0, max(err_norm), 40))
    ax1.legend(['Error norm'], loc='upper center', bbox_to_anchor=(0.80, 0.86),
          fancybox=True, shadow=True, ncol=1, fontsize=12)    
    
    # Mark changes in 'current_goal_set' on the x-axis for 'err_data'
    for change in changes_err:
        ax1.axvline(x=change, color='k', linestyle='--', linewidth=0.6)
    
    # Adding custom ticks at changes, including start and end of the data
    all_ticks_err = np.concatenate(([0], changes_err, [len(current_goal_sets_err)-1]))
    ax1.set_xticks(all_ticks_err)
    ax1.set_xticklabels([str(int(current_goal_sets_err[tick])) for tick in all_ticks_err])
    
    # Plot individual feature errors  
    for col in range(len(err_data[0])):
        ax2.plot(err_data[1:, col], linewidth=1.6)   
    ax2.margins(x = 0.0, y=0.0)
    ax2.set_ylabel('Feature errors (px)', fontsize=10)
    ax2.set_xlabel('Control loop iteration #', fontsize=10)
    ax2.grid()
    # ax2.set_xticks(xtick_values)
    # ax2.set_xticklabels(xtick_labels)
    ax2.set_xticks(np.arange(0, len(err_norm), 40))
    # ax2.set_yticks(np.arange(0, max(err_norm), 26))
    ax2.legend(['x1','y1','x2','y2', 'x3', 'y3', 'x4', 'y4'],loc='upper center',
    bbox_to_anchor=(0.6, -0.30), fancybox=True, shadow=True, ncol=len(err_data[0]), fontsize=8)
    
    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/feature_error.png', dpi=300)
    # Image.open('feature_error.png').convert("RGB").save('feature_error.jpg','JPEG')

    # plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/8/feature_error.png', dpi=400)
    # Image.open('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/8/feature_error.png').convert("RGB").save('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/8/feature_error.jpg','JPEG')


    model_error = []
    j1_vel = []
    j2_vel = []
    j3_vel = []
    # Read the model error
    # with open('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/8/modelerror.csv','r') as csvfile:
    with open('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/modelerror.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        for row in plots:
            model_error.append(row)

    # Read joint velocities
    # with open('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/8/j1vel.csv','r') as csvfile:
    with open('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/j1vel.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        for row in plots:
            j1_vel.append(row)
    
    # with open('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/8/j2vel.csv','r') as csvfile:
    with open('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/j2vel.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        for row in plots:
            j2_vel.append(row)

    # # with open('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps/8/j3vel.csv','r') as csvfile:
    with open('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/j3vel.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        for row in plots:
            j3_vel.append(row)

    # Post process the list
    model_error = model_error[1:]
    model_error = [list( map(float,i) ) for i in model_error]
    model_error = np.array(model_error)
    current_goal_sets = model_error[:, 0]  # Extract the first column for current_goal_set
    model_error = model_error[:, 1:]  # Skip the first column for the actual model error data

    changes_model = np.where(np.diff(model_error[:, 0]) != 0)[0] + 1

    for change in changes_model:
        ax1.axvline(x=change, color='k', linestyle='--', linewidth=0.6)

    j1_vel = j1_vel[1:]
    j1_vel = [list( map(float,i) ) for i in j1_vel]
    j1_vel = np.array(j1_vel)[:, 1:]

    j2_vel = j2_vel[1:]
    j2_vel = [list( map(float,i) ) for i in j2_vel]
    j2_vel = np.array(j2_vel)[:, 1:]

    j3_vel = j3_vel[1:]
    j3_vel = [list( map(float,i) ) for i in j3_vel]
    j3_vel = np.array(j3_vel)[:, 1:]

    # Create axes for plots
    fig, axes = plt.subplots(nrows=1, ncols=1)
    ax1 = axes
    # ax2 = axes[1]
    # ax3 = axes[2]
    # ax4 = axes[3]
    fig.tight_layout(pad=4.0)

    ax1.plot(model_error, 'b',linewidth=1.7)
    # Mark changes in current_goal_set on the x-axis
    # for goal_set_change_idx in unique_goal_sets:
    #     ax1.axvline(x=goal_set_change_idx, color='k', linestyle='--', linewidth=0.6)
    ax1.margins(x = 0.0, y=0.0)
    ax1.set_ylabel('Model error', fontsize=10)
    ax1.set_xlabel('Iteration #', fontsize=10)
    ax1.grid()
    ax1.set_xticks(np.arange(0, len(model_error), 80))
    # ax1.set_xticks(xtick_values)
    # ax1.set_xticklabels(xtick_labels)
    # ax1.set_yticks(np.arange(0, max(model_error), 400))
    ax1.legend(['Model error'], loc='upper center', bbox_to_anchor=(0.80, 0.90),
          fancybox=True, shadow=True, ncol=1, fontsize=10)

    # plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/8/model_error.png', dpi=400)
    # Image.open('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/8/model_error.png').convert("RGB").save('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/8/model_error.jpg','JPEG')

    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/model_error.png', dpi=300)
    # Image.open('model_error.png').convert("RGB").save('model_error.jpg','JPEG')
    
    fig, axes = plt.subplots(nrows=3, ncols=1)
    ax2 = axes[0]
    ax3 = axes[1]
    ax4 = axes[2]
    fig.tight_layout(pad=4.0)
    ax2.plot(j1_vel, 'b',linewidth=1.6)
    # Mark changes in current_goal_set
    # for goal_set_change_idx in unique_goal_sets:
    #     ax2.axvline(x=goal_set_change_idx, color='k', linestyle='--', linewidth=0.6)
    ax2.margins(x = 0.0, y=0.0)
    ax2.set_ylim(-0.6,0.6)
    ax2.set_ylabel('J1 Velocity (rad/s)', fontsize=10)
    ax2.set_xlabel('Iteration #', fontsize=10)
    ax2.grid()
    ax2.set_xticks(np.arange(0, len(model_error), 26))
    # ax2.legend(['Joint 1'], loc='upper center', bbox_to_anchor=(0.80, 0.90),
    #       fancybox=True, shadow=True, ncol=1, fontsize=11)

    ax3.plot(j2_vel, 'b',linewidth=1.6)
    # Mark changes in current_goal_set
    # for goal_set_change_idx in unique_goal_sets:
    #     ax3.axvline(x=goal_set_change_idx, color='k', linestyle='--', linewidth=0.6)
    ax3.margins(x = 0.0, y=0.0)
    ax3.set_ylim(-0.6,0.6)
    ax3.set_ylabel('J2 Velocity (rad/s)', fontsize=10)
    ax3.set_xlabel('Iteration #', fontsize=10)
    ax3.grid()
    ax3.set_xticks(np.arange(0, len(model_error), 26))
    # ax3.legend(['Joint 2'], loc='upper center', bbox_to_anchor=(0.80, 0.90),
    #       fancybox=True, shadow=True, ncol=1, fontsize=11)

    ax4.plot(j3_vel, 'b',linewidth=1.6)
    # Mark changes in current_goal_set
    # for goal_set_change_idx in unique_goal_sets:
    #     ax4.axvline(x=goal_set_change_idx, color='k', linestyle='--', linewidth=0.6)
    ax4.margins(x = 0.0, y=0.0)
    ax4.set_ylim(-0.6,0.6)
    ax4.set_ylabel('J3 Velocity (rad/s)')
    ax4.set_xlabel('Iteration #')
    ax4.grid()
    ax4.set_xticks(np.arange(0, len(model_error), 26))
    # ax4.legend(['Joint 3'], loc='upper center', bbox_to_anchor=(0.6, -.76),
    #       fancybox=True, shadow=True, ncol=1)


    plt.savefig('plot.png', dpi=300)
    Image.open('plot.png').convert("RGB").save('plot.jpg','JPEG')

    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/plot.png', dpi=400)
    # Image.open('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/8/plot.png').convert("RGB").save('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/8/plot.jpg','JPEG')

    # Joint 1 Velocity Plot
    plt.figure(figsize=(10, 6))
    plt.plot(j1_vel, 'b', linewidth=1.6)
    plt.margins(x=0.0, y=0.0)
    plt.ylim(-0.6, 0.6)
    plt.ylabel('J1 Velocity (rad/s)', fontsize=10)
    plt.xlabel('Iteration #', fontsize=10)
    plt.grid()
    plt.xticks(np.arange(0, len(j1_vel), 26))
    plt.title('Joint 1 Velocity over Time')
    plt.tight_layout()
    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/j1_velocity.png', dpi=300)
    plt.close()

    # Joint 2 Velocity Plot
    plt.figure(figsize=(10, 6))
    plt.plot(j2_vel, 'b', linewidth=1.6)
    plt.margins(x=0.0, y=0.0)
    plt.ylim(-0.6, 0.6)
    plt.ylabel('J2 Velocity (rad/s)', fontsize=10)
    plt.xlabel('Iteration #', fontsize=10)
    plt.grid()
    plt.xticks(np.arange(0, len(j2_vel), 26))
    plt.title('Joint 2 Velocity over Time')
    plt.tight_layout()
    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/j2_velocity.png', dpi=300)
    plt.close()

    # Joint 3 Velocity Plot
    plt.figure(figsize=(10, 6))
    plt.plot(j3_vel, 'b', linewidth=1.6)
    plt.margins(x=0.0, y=0.0)
    plt.ylim(-0.6, 0.6)
    plt.ylabel('J3 Velocity (rad/s)')
    plt.xlabel('Iteration #')
    plt.grid()
    plt.xticks(np.arange(0, len(j3_vel), 26))
    plt.title('Joint 3 Velocity over Time')
    plt.tight_layout()
    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/8/j3_velocity.png', dpi=300)
    plt.close()




    print("Plotting complete")

    # rospy.signal_shutdown("Shutting down")


if __name__ == "__main__":
    main(sys.argv)
