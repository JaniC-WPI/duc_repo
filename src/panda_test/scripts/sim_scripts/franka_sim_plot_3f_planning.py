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

    err_data = []
    # read feature error
    with open('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/14_a/err.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            err_data.append(row)

    # Post process the list
    err_data = err_data[1:]
    err_data = [list(map(float, i)) for i in err_data]
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

    # Adding ticks at every 40 iterations and custom ticks at changes
    iteration_ticks = np.arange(0, len(err_norm), 100)
    all_ticks_err = np.unique(np.concatenate((iteration_ticks, changes_err, [len(current_goal_sets_err) - 1])))

    # Create custom tick labels
    tick_labels = ['' for _ in range(len(all_ticks_err))]
    goal_labels = ['' for _ in range(len(all_ticks_err))]
    iteration_labels = ['' for _ in range(len(all_ticks_err))]

    for idx, tick in enumerate(all_ticks_err):
        if tick in changes_err:
            goal_labels[idx] = f'{int(current_goal_sets_err[tick])}'
        iteration_labels[idx] = f'{tick}'

    # Plot error norm
        print(max(err_norm))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(top=0.95)
    ax1.plot(err_norm, 'b', linewidth=1.6)
    ax1.margins(x=0.0, y=0.0)
    # ax1.set_ylabel('Error norm', fontsize=10)
    ax1.grid(False)
    ax1.set_yticks(np.arange(0, max(err_norm)+40, 40))
    # ax1.tick_params(axis='y', labelsize=6)  # Reduce y-axis tick size
    # ax1.legend(['Error norm'], loc='upper center', bbox_to_anchor=(0.80, 0.86),
    #            fancybox=True, shadow=True, ncol=1, fontsize=12)

    # Mark changes in 'current_goal_set' on the x-axis for 'err_data'
    for change in changes_err:
        ax1.axvline(x=change, color='k', linestyle='--', linewidth=1.0)

    # Set specific x-tick labels with larger, bold font
    ax1.set_xticks(all_ticks_err)
    ax1.set_xticklabels(goal_labels, fontsize=14, weight='bold')

    # Set y-tick labels with larger, bold font
    ax1.tick_params(axis='y', labelsize=14)
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')

    secax1 = ax1.secondary_xaxis('top')
    # secax1.set_xlabel('Control loop iteration #', fontsize=10)
    secax1.set_xticks([])
    secax1.set_xticklabels([])

    secax2 = ax1.secondary_xaxis(-0.075)
    # secax2.set_xticks(all_ticks_err)
    # secax2.set_xticklabels(iteration_labels, fontsize=6)

    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/14_a/error_norm.png', dpi=300)
    plt.close()

    # Plot individual feature errors
    fig, ax2 = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(top=0.95)
    for col in range(len(err_data[0])):
        ax2.plot(err_data[1:, col], linewidth=1.6)
    ax2.margins(x=0.0, y=0.0)
    ax2.set_ylabel('Feature errors (px)', fontsize=10)
    ax2.grid()
    yticks = np.arange(-60, 200, 20)
    yticks = np.append(yticks, 0)
    yticks = np.sort(yticks)
    ax2.set_yticks(yticks)
    ax2.tick_params(axis='y', labelsize=6)  # Reduce y-axis tick size
    ax2.legend(['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6'], loc='upper center',
               bbox_to_anchor=(0.5, -0.08), fancybox=True, shadow=True, ncol=len(err_data[0]), fontsize=6)

    ax2.set_xticks(all_ticks_err)
    ax2.set_xticklabels(goal_labels, fontsize=6)
    ax2.tick_params(axis='x', pad=1)

    secax1 = ax2.secondary_xaxis('top')
    secax1.set_xlabel('Control loop iteration #', fontsize=10)
    secax1.set_xticks([])
    secax1.set_xticklabels([])

    secax2 = ax2.secondary_xaxis(-0.04)
    secax2.set_xticks(all_ticks_err)
    secax2.set_xticklabels(iteration_labels, fontsize=6)

    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/14_a/feature_errors.png', dpi=300)
    plt.close()

    mod_err_data = []
    # read feature error
    with open('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/14_a/individual_model_errors.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        headers = next(csv_reader)  # Assuming the first row is headers
        for row in csv_reader:
            mod_err_data.append([float(val) for val in row])

    mod_err_data = np.array(mod_err_data)

    # # Check if the array contains any NaNs
    # if np.isnan(mod_err_data).any():
    #     print("NaN values detected in mod_err_data")

    # Plot the data
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    fig.tight_layout(pad=4.0)
    fig.subplots_adjust(top=0.95)

    # Plot individual model errors
    for col in range(mod_err_data.shape[1]):
        ax1.plot(mod_err_data[:, col], linewidth=1.6)

    ax1.margins(x=0.0, y=0.0)
    ax1.set_ylabel('Feature model errors (px)', fontsize=10)
    ax1.grid()
    # ax1.set_yticks(np.arange(0, len(mod_err_data)), 200)
    ax1.tick_params(axis='y', labelsize=6)  # Reduce y-axis tick size
    ax1.legend(['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5'], loc='lower center',
               bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=len(mod_err_data[0]), fontsize=6)

    # # Mark changes in 'current_goal_set' on the x-axis for 'err_data'
    for change in changes_err:
        ax1.axvline(x=change, color='k', linestyle='--', linewidth=0.6)

    # # Adding ticks at every 40 iterations and custom ticks at changes
    ax1.set_xticks(all_ticks_err)
    ax1.set_xticklabels(goal_labels, fontsize=6)
    ax1.tick_params(axis='x', pad=1)

    secax1 = ax1.secondary_xaxis('top')
    secax1.set_xlabel('Control loop iteration #', fontsize=10)
    secax1.set_xticks([])
    secax1.set_xticklabels([])

    secax2 = ax1.secondary_xaxis(-0.04)
    secax2.set_xticks(all_ticks_err)
    secax2.set_xticklabels(iteration_labels, fontsize=6)

    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/14_a/ind_mod_error.png', dpi=300)
    plt.close()

    model_error = []
    j1_vel = []
    j2_vel = []
    j3_vel = []
    # Read the model error
    with open('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/14_a/modelerror.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            model_error.append(row)

    # # Read joint velocities
    with open('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/14_a/j1vel.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            j1_vel.append(row)

    with open('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/14_a/j2vel.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            j2_vel.append(row)

    with open('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/14_a/j3vel.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            j3_vel.append(row)

    # # Post process the list
    model_error = model_error[1:]
    model_error = [list(map(float, i)) for i in model_error]
    model_error = np.array(model_error)
    current_goal_sets = model_error[:, 0]  # Extract the first column for current_goal_set
    model_error = model_error[:, 1:]  # Skip the first column for the actual model error data

    changes_model = np.where(np.diff(current_goal_sets) != 0)[0] + 1

    j1_vel = j1_vel[1:]
    j1_vel = [list(map(float, i)) for i in j1_vel]
    j1_vel = np.array(j1_vel)[:, 1:]

    j2_vel = j2_vel[1:]
    j2_vel = [list(map(float, i)) for i in j2_vel]
    j2_vel = np.array(j2_vel)[:, 1:]

    j3_vel = j3_vel[1:]
    j3_vel = [list(map(float, i)) for i in j3_vel]
    j3_vel = np.array(j3_vel)[:, 1:]

    # Create axes for plots
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.tight_layout(pad=4.0)
    fig.subplots_adjust(top=0.95)

    ax1.plot(model_error, '#271AA5', linewidth=2.0)
    ax1.margins(x=0.0, y=0.0)
    ax1.set_ylim(min(model_error) - 500, max(model_error) + 750)
    # ax1.set_ylabel('Model error', fontsize=10)
    ax1.grid(False)
    # ax1.set_xticks(np.arange(0, len(model_error)))
    # ax1.set_yticks(np.arange(-500, max(model_error))+1000)
    # ax1.tick_params(axis='y', labelsize=6)  # Reduce y-axis tick size
    # # ax1.legend(['Model error'], loc='upper center', bbox_to_anchor=(0.80, 0.90),
    # #            fancybox=True, shadow=True, ncol=1, fontsize=10)

    # # Mark changes in 'current_goal_set' on the x-axis for 'model_error'
    for change in changes_err:
        ax1.axvline(x=change, color='k', linestyle='--', linewidth=1.0)

    # # Adding ticks at every 40 iterations and custom ticks at changes  

    ax1.set_xticks(all_ticks_err)
    ax1.set_xticklabels(goal_labels, fontsize=14, weight='bold')

    ax1.tick_params(axis='y', labelsize=14)
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')

    secax1 = ax1.secondary_xaxis('top')
    # secax1.set_xlabel('Control loop iteration #', fontsize=10)
    secax1.set_xticks([])
    secax1.set_xticklabels([])

    secax2 = ax1.secondary_xaxis(-0.075)

    # secax2.set_xticks(all_ticks_err)
    # secax2.set_xticklabels(iteration_labels, fontsize=6)

    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/14_a/model_error.svg', dpi=300)
    plt.close()

    # fig, axes = plt.subplots(nrows=3, ncols=1)
    # ax2 = axes[0]
    # ax3 = axes[1]
    # ax4 = axes[2]
    # fig.tight_layout(pad=4.0)

    # ax2.plot(j1_vel, 'b', linewidth=1.6)
    # ax2.margins(x=0.0, y=0.0)
    # ax2.set_ylim(-0.6, 0.6)
    # ax2.set_ylabel('J1 Velocity (rad/s)', fontsize=10)
    # ax2.grid()
    # ax2.set_xticks(np.arange(0, len(model_error), 26))
    # ax2.tick_params(axis='y', labelsize=8)  # Reduce y-axis tick size

    # ax3.plot(j2_vel, 'b', linewidth=1.6)
    # ax3.margins(x=0.0, y=0.0)
    # ax3.set_ylim(-0.6, 0.6)
    # ax3.set_ylabel('J2 Velocity (rad/s)', fontsize=10)
    # ax3.grid()
    # ax3.set_xticks(np.arange(0, len(model_error), 26))
    # ax3.tick_params(axis='y', labelsize=8)  # Reduce y-axis tick size

    # ax4.plot(j3_vel, 'b', linewidth=1.6)
    # ax4.margins(x=0.0, y=0.0)
    # ax4.set_ylim(-0.6, 0.6)
    # ax4.set_ylabel('J3 Velocity (rad/s)')
    # ax4.grid()
    # ax4.set_xticks(np.arange(0, len(model_error), 26))
    # ax4.tick_params(axis='y', labelsize=8)  # Reduce y-axis tick size

    # # Mark changes in 'current_goal_set' on the x-axis for 'j1_vel', 'j2_vel', and 'j3_vel'
    # for change in changes_err:
    #     ax2.axvline(x=change, color='k', linestyle='--', linewidth=0.6)
    #     ax3.axvline(x=change, color='k', linestyle='--', linewidth=0.6)
    #     ax4.axvline(x=change, color='k', linestyle='--', linewidth=0.6)

    # # Adding ticks at every 40 iterations and custom ticks at changes
    # for ax in [ax2, ax3, ax4]:
    #     ax.set_xticks(all_ticks_err)
    #     ax.set_xticklabels(goal_labels, fontsize=8)
    #     ax.tick_params(axis='x', pad=15)

    #     secax1 = ax.secondary_xaxis('top')
    #     secax1.set_xlabel('Control loop iteration #', fontsize=10)
    #     secax1.set_xticks([])
    #     secax1.set_xticklabels([])

    #     secax2 = ax.secondary_xaxis('bottom')
    #     secax2.set_xticks(all_ticks_err)
    #     secax2.set_xticklabels(iteration_labels, fontsize=8)

    # plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/14_a/plot.png', dpi=400)
    # plt.close()

    all_velocities = j1_vel + j2_vel + j3_vel  # Combine all velocity lists
    # print(all_velocities)
    min_j1 = min(j1_vel)
    max_j1 = max(j1_vel)
    min_j2 = min(j2_vel)
    max_j2 = max(j2_vel)
    min_j3 = min(j3_vel)
    max_j3 = max(j3_vel)
    print(min_j1,max_j1, min_j2, max_j2, min_j3, max_j3)
    global_min = min(min_j1, min_j2, min_j3) - 0.01  # Adjust as needed
    global_max = max(max_j1, max_j2, max_j3) + 0.01 # Adjust as needed
    print(global_min)
    print(global_max)

    # Joint 1 Velocity Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(top=0.95)
    ax.plot(j1_vel, 'b', linewidth=1.6)
    ax.margins(x=0.0, y=0.0)
    ax.set_ylim(global_min, global_max)
    # ax.set_ylabel('J1 Velocity (rad/s)', fontsize=10)
    ax.grid(False)
    for change in changes_err:
        ax.axvline(x=change, color='k', linestyle='--', linewidth=1.0)
    # ax.tick_params(axis='y', labelsize=6)  # Reduce y-axis tick size
    # Set tick labels with larger, bold font
    ax.tick_params(axis='x', labelsize=1, pad=1)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    ax.set_xticks(all_ticks_err)
    ax.set_xticklabels(goal_labels, fontsize=14, weight='bold')

    # Set y-tick labels with larger, bold font
    ax.tick_params(axis='y', labelsize=14)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    secax1 = ax.secondary_xaxis('top')
    # secax1.set_xlabel('Control loop iteration #', fontsize=10)
    secax1.set_xticks([])
    secax1.set_xticklabels([])

    secax2 = ax.secondary_xaxis(-0.075)
    # secax2.set_xticks(all_ticks_err)
    # secax2.set_xticklabels(iteration_labels, fontsize=6)

    # ax.set_title('Joint 1 Velocity over Time')
    # plt.tight_layout()
    # plt.grid(False)
    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/14_a/j1_velocity.png', dpi=300)
    plt.close()

    # # Joint 2 Velocity Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(top=0.95)
    ax.plot(j2_vel, 'b', linewidth=1.6)
    ax.margins(x=0.0, y=0.0)
    ax.set_ylim(global_min, global_max)
    # ax.set_ylabel('J2 Velocity (rad/s)', fontsize=10)
    ax.grid(False)

    for change in changes_err:
        ax.axvline(x=change, color='k', linestyle='--', linewidth=1.0)

    # Set tick labels with larger, bold font
    ax.tick_params(axis='x', labelsize=1, pad=1)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    ax.set_xticks(all_ticks_err)
    ax.set_xticklabels(goal_labels, fontsize=14, weight='bold')

    # Set y-tick labels with larger, bold font
    ax.tick_params(axis='y', labelsize=14)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    secax1 = ax.secondary_xaxis('top')
    # secax1.set_xlabel('Control loop iteration #', fontsize=10)
    secax1.set_xticks([])
    secax1.set_xticklabels([])

    secax2 = ax.secondary_xaxis(-0.075)
    # secax2.set_xticks(all_ticks_err)
    # secax2.set_xticklabels(iteration_labels, fontsize=6)

    # ax.set_title('Joint 2 Velocity over Time')
    # plt.tight_layout()
    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/14_a/j2_velocity.png', dpi=300)
    plt.close()

    # # Joint 3 Velocity Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(top=0.95)
    ax.plot(j3_vel, 'b', linewidth=1.6)
    ax.margins(x=0.0, y=0.0)
    ax.set_ylim(global_min, global_max)
    # ax.set_ylabel('J3 Velocity (rad/s)')
    ax.grid(False)

    for change in changes_err:
        ax.axvline(x=change, color='k', linestyle='--', linewidth=1.0)

    # Set tick labels with larger, bold font
    ax.tick_params(axis='x', labelsize=1, pad=1)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    ax.set_xticks(all_ticks_err)
    ax.set_xticklabels(goal_labels, fontsize=14, weight='bold')

    # Set y-tick labels with larger, bold font
    ax.tick_params(axis='y', labelsize=14)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    secax1 = ax.secondary_xaxis('top')
    # secax1.set_xlabel('Control loop iteration #', fontsize=10)
    secax1.set_xticks([])
    secax1.set_xticklabels([])

    secax2 = ax.secondary_xaxis(-0.075)
    # secax2.set_xticks(all_ticks_err)
    # secax2.set_xticklabels(iteration_labels, fontsize=)

    # ax.set_title('Joint 3 Velocity over Time')
    # plt.tight_layout()
    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/14_a/j3_velocity.png', dpi=300)
    plt.close()

    # Combined Joint Velocities Plot
    # Combined Joint Velocities Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(top=0.95)

    # Plot each joint velocity in a different color
    ax.plot(j1_vel, color='#271AA5', linewidth=2, label='Joint 2')
    ax.plot(j2_vel, color='#16720C', linewidth=2, label='Joint 4')
    ax.plot(j3_vel, color='#D21B07', linewidth=2, label='Joint 6')

    # Adjust plot limits and labels
    ax.margins(x=0.0, y=0.0)
    ax.set_ylim(global_min, global_max)
    # ax.set_ylabel('Velocity (rad/s)', fontsize=10)
    ax.grid(False)

    # Mark changes in 'current_goal_set' on the x-axis
    for change in changes_err:
        ax.axvline(x=change, color='k', linestyle='--', linewidth=1.0)

    # Set specific x-tick labels with larger, bold font for goal labels
    # ax.set_xticks(all_ticks_err)
    # ax.set_xticklabels(goal_labels, fontsize=14, weight='bold')
    # ax.tick_params(axis='x', labelsize=1, pad=1)

    ax.set_xticks(all_ticks_err)
    ax.set_xticklabels(goal_labels, fontsize=16, weight='bold')
    # Set y-tick labels with larger, bold font
    ax.tick_params(axis='y', labelsize=16)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # Add secondary x-axis
    secax1 = ax.secondary_xaxis('top')
    secax1.set_xticks([])
    secax1.set_xticklabels([])

    secax2 = ax.secondary_xaxis(-0.075)

    # Add legend with bold font using 'prop' argument
    ax.legend(loc='upper right', prop={'size':18, 'weight': 'bold'})

    # Save the plot
    plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/sim_vs/servoing/exps/14_a/combined_joint_velocities.svg', dpi=300)
    plt.close()

    print("Plotting complete")

    # rospy.signal_shutdown("Shutting down")

if __name__ == "__main__":
    main(sys.argv)
