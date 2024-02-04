#!/bin/bash
mkdir ~/.ros/raw
# mkdir ~/.ros/d_raw
cd ~/duc_repo

# roslaunch encoderless_vs franka_dream_feature.launch i:=/home/merlab/DREAM/trained_models/panda_dream_resnet_h.pth b:=panda_link0 &&

roslaunch panda_test franka_sim_goal_feature.launch
# roslaunch encoderless_vs franka_dream_service.launch i:=/home/merlab/DREAM/trained_models/panda_dream_resnet_h.pth b:=panda_link0

mv ~/.ros/dl_features.yaml ~/duc_repo/src/panda_test/config/

# Renaming the dir
mv ~/.ros/raw ~/.ros/goal_raw
# mv ~/.ros/d_raw ~/.ros/goal_d_raw