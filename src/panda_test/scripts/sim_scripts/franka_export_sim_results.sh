#!/bin/bash

read exp_no

cd ~/Pictures/Dl_Exps/sim_vs/servoing/exps/ 

mkdir $exp_no

echo $dir_path

mv ~/.ros/dr.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/ds.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/err.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/j1vel.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/j2vel.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/j3vel.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/cp.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/modelerror.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/individual_model_errors.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/qhat.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/qhat_feat.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/joint_angles.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/servoing_error.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/servoing_error.bag ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/current_goal_set_topic.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/current_goal_set_topic.bag ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/dr_record.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/dr_record.bag ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/ds_record.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/ds_record.bag ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/J_modelerror.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/J_modelerror.bag ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/joint_vel.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/joint_vel.bag ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/vsbot_control_points.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/vsbot_control_points.bag ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/vsbot_status.csv ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/vsbot_status.bag ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/*.jpg ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mv ~/.ros/exp_vid.avi ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

mkdir ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/imgs

mv ~/.ros/*.png ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/imgs/

mv ~/.ros/raw ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/
mv ~/.ros/goal_raw ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/
# mv ~/.ros/goal_d_raw ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/
# mv ~/.ros/d_raw ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/

cp ~/duc_repo/src/panda_test/config/franka_sim_config.yaml  ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/
cp ~/duc_repo/src/panda_test/config/dl_multi_features.yaml  ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/


# cp ~/duc_repo/src/panda_test/launch/franka_sim.launch  ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/
# cp ~/duc_repo/src/panda_test/launch/franka_sim_goal.launch  ~/Pictures/Dl_Exps/sim_vs/servoing/exps/$exp_no/


