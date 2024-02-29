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
    # with open('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/4/err.csv','r') as csvfile:
    with open('err.csv','r') as csvfile:
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

    # create plot axes
    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax1 = axes[0]
    ax2 = axes[1]
    fig.tight_layout(pad=4.0)

    

    # Plot error norm   
    ax1.plot(err_norm, 'b',linewidth=1.5)
    ax1.margins(x = 0.0, y=0.0)
    ax1.set_ylabel('Error norm', fontsize=13)
    ax1.set_xlabel('Control loop iteration #', fontsize=13)
    ax1.grid()    
    ax1.set_yticks(np.arange(0, max(err_norm), 25))
    ax1.legend(['Error norm'], loc='upper center', bbox_to_anchor=(0.80, 0.85),
          fancybox=True, shadow=True, ncol=1, fontsize=12)    
    
    # Plot individual feature errors  
    for col in range(len(err_data[0])):
        ax2.plot(err_data[1:, col], linewidth=1.5)   
    ax2.margins(x = 0.0, y=0.0)
    ax2.set_ylabel('Feature errors (px)', fontsize=13)
    ax2.set_xlabel('Control loop iteration #', fontsize=13)
    ax2.grid()
    # ax2.set_xticks(xtick_values)
    # ax2.set_xticklabels(xtick_labels)
    ax2.set_xticks(np.arange(0, len(err_norm), 20))
    # ax2.set_yticks(np.arange(0, max(err_norm), 25))
    ax2.legend(['x1','y1','x2','y2', 'x3', 'y3', 'x4', 'y4'],loc='upper center',
    bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=len(err_data[0]), fontsize=12)
    
    plt.savefig('feature_error.png', dpi=300)
    Image.open('feature_error.png').convert("RGB").save('feature_error.jpg','JPEG')

    # plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/4/feature_error.png', dpi=400)
    # Image.open('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/4/feature_error.png').convert("RGB").save('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/4/feature_error.jpg','JPEG')


    model_error = []
    j1_vel = []
    j2_vel = []
    j3_vel = []
    # Read the model error
    # with open('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/4/modelerror.csv','r') as csvfile:
    with open('modelerror.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        for row in plots:
            model_error.append(row)

    # Read joint velocities
    # with open('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/4/j1vel.csv','r') as csvfile:
    with open('j1vel.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        for row in plots:
            j1_vel.append(row)
    
    # with open('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/4/j2vel.csv','r') as csvfile:
    with open('j2vel.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        for row in plots:
            j2_vel.append(row)

    # # with open('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps/5/j3vel.csv','r') as csvfile:
    with open('j3vel.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        for row in plots:
            j3_vel.append(row)

    # Post process the list
    model_error = model_error[1:]
    model_error = [list( map(float,i) ) for i in model_error]
    model_error = np.array(model_error)

    j1_vel = j1_vel[1:]
    j1_vel = [list( map(float,i) ) for i in j1_vel]
    j1_vel = np.array(j1_vel)

    j2_vel = j2_vel[1:]
    j2_vel = [list( map(float,i) ) for i in j2_vel]
    j2_vel = np.array(j2_vel)

    j3_vel = j3_vel[1:]
    j3_vel = [list( map(float,i) ) for i in j3_vel]
    j3_vel = np.array(j3_vel)

    # Create axes for plots
    fig, axes = plt.subplots(nrows=1, ncols=1)
    ax1 = axes
    # ax2 = axes[1]
    # ax3 = axes[2]
    # ax4 = axes[3]
    fig.tight_layout(pad=4.0)

    ax1.plot(model_error, 'b',linewidth=1.5)
    ax1.margins(x = 0.0, y=0.0)
    ax1.set_ylabel('Model error', fontsize=15)
    ax1.set_xlabel('Iteration #', fontsize=13)
    ax1.grid()
    ax1.set_xticks(np.arange(0, len(model_error), 20))
    # ax1.set_xticks(xtick_values)
    # ax1.set_xticklabels(xtick_labels)
    # ax1.set_yticks(np.arange(0, max(model_error), 400))
    ax1.legend(['Model error'], loc='upper center', bbox_to_anchor=(0.80, 0.90),
          fancybox=True, shadow=True, ncol=1, fontsize=11)

    # plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/4/model_error.png', dpi=400)
    # Image.open('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/4/model_error.png').convert("RGB").save('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/4/model_error.jpg','JPEG')

    plt.savefig('model_error.png', dpi=300)
    Image.open('model_error.png').convert("RGB").save('model_error.jpg','JPEG')
    
    fig, axes = plt.subplots(nrows=3, ncols=1)
    ax2 = axes[0]
    ax3 = axes[1]
    ax4 = axes[2]
    fig.tight_layout(pad=4.0)
    ax2.plot(j1_vel, 'b',linewidth=1.5)
    ax2.margins(x = 0.0, y=0.0)
    ax2.set_ylim(-0.1,0.1)
    ax2.set_ylabel('Velocity (rad/s)', fontsize=13)
    ax2.set_xlabel('Iteration #', fontsize=13)
    ax2.grid()
    ax2.set_xticks(np.arange(0, len(model_error), 20))
    # ax2.legend(['Joint 1'], loc='upper center', bbox_to_anchor=(0.80, 0.90),
    #       fancybox=True, shadow=True, ncol=1, fontsize=11)

    ax3.plot(j2_vel, 'b',linewidth=1.5)
    ax3.margins(x = 0.0, y=0.0)
    ax3.set_ylim(-0.1,0.1)
    ax3.set_ylabel('Velocity (rad/s)', fontsize=13)
    ax3.set_xlabel('Iteration #', fontsize=13)
    ax3.grid()
    ax3.set_xticks(np.arange(0, len(model_error), 20))
    # ax3.legend(['Joint 2'], loc='upper center', bbox_to_anchor=(0.80, 0.90),
    #       fancybox=True, shadow=True, ncol=1, fontsize=11)

    ax4.plot(j3_vel, 'b',linewidth=1.5)
    ax4.margins(x = 0.0, y=0.0)
    ax4.set_ylim(-0.1,0.1)
    ax4.set_ylabel('Velocity (rad/s)')
    ax4.set_xlabel('Iteration #')
    ax4.grid()
    ax4.set_xticks(np.arange(0, len(model_error), 25))
    # ax4.legend(['Joint 3'], loc='upper center', bbox_to_anchor=(0.5, -.75),
    #       fancybox=True, shadow=True, ncol=1)


    plt.savefig('plot.png', dpi=300)
    Image.open('plot.png').convert("RGB").save('plot.jpg','JPEG')

    # plt.savefig('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/4/plot.png', dpi=400)
    # Image.open('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/4/plot.png').convert("RGB").save('/home/jc-merlab/Pictures/Dl_Exps/dl_vs/servoing/exps_latest_planar_02_21_2023/4/plot.jpg','JPEG')


    print("Plotting complete")

    # rospy.signal_shutdown("Shutting down")


if __name__ == "__main__":
    main(sys.argv)
