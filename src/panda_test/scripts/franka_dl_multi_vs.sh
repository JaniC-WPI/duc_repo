#!/bin/bash
mkdir ~/.ros/raw &&
# mkdir ~/.ros/d_raw &&
cd ~/duc_repo

source devel/setup.bash

# roslaunch panda_test franka_sim.launch
roslaunch panda_test franka_dl_multi_vs.launch &&
# roslaunch encoderless_vs franka_shape_multiref_vs.launch &&
roslaunch panda_test franka_dl_multi_plot.launch &&

cd ~/duc_repo/src/panda_test/scripts/

./franka_export_multi_dl_results.sh
