#include "ros/ros.h"
#include <iostream>
#include <fstream>
#include <string>

#include "sensor_msgs/JointState.h"
#include "std_msgs/Int32.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"

int status = 0;
int current_goal_set = 0;

int no_of_features;

// n.getParam("vsbot/shape_control/no_of_features", no_of_features);

int no_of_actuators;

// n.getParam("vsbot/shape_control/no_of_features", no_of_actuators);

void statusCallback(const std_msgs::Int32 &status_msg){
    status = status_msg.data;
}

// void updateGoalSetIndex(const std::vector<float>& error) {
//     // Implement logic to determine if the goal set should be updated
//     // This might be based on the error magnitude or some other criteria
//     // For example:
//     float error_norm = std::sqrt(std::inner_product(error.begin(), error.end(), error.begin(), 0.0));
//     if (error_norm < 30) {
//         current_goal_set++;
//         // Reset or adjust other relevant variables if needed
//     }
// }

void currentGoalSetCallback(const std_msgs::Int32 &msg) {
    current_goal_set = msg.data;
}

void dsCallback(const std_msgs::Float64MultiArray &msg){
    // Write ds to excel
    std::ofstream ds_plotdata("ds.csv",std::ios::app);
    // uncomment next line for 3 features
    if (no_of_features==6){
        ds_plotdata<<current_goal_set<<","<<msg.data.at(0)<<","<<msg.data.at(1)<<","<<msg.data.at(2)<<","
        <<msg.data.at(3)<<","<<msg.data.at(4)<<","<<msg.data.at(5)<<"\n";
    }
    else if (no_of_features==8){
        ds_plotdata<<current_goal_set<<","<<msg.data.at(0)<<","<<msg.data.at(1)<<","<<msg.data.at(2)<<","<<msg.data.at(3)<<","
        <<msg.data.at(4)<<","<<msg.data.at(5)<<","<<msg.data.at(6)<<","<<msg.data.at(7)<<"\n";
    }  
    else if (no_of_features==10){
        ds_plotdata<<current_goal_set<<","<<msg.data.at(0)<<","<<msg.data.at(1)<<","<<msg.data.at(2)<<","<<msg.data.at(3)<<","
        <<msg.data.at(4)<<","<<msg.data.at(5)<<","<<msg.data.at(6)<<","<<msg.data.at(7)<<","
        <<msg.data.at(8)<<","<<msg.data.at(9)<<"\n";
    }  
    else if (no_of_features==12){
        ds_plotdata<<current_goal_set<<","<<msg.data.at(0)<<","<<msg.data.at(1)<<","<<msg.data.at(2)<<","<<msg.data.at(3)<<","
        <<msg.data.at(4)<<","<<msg.data.at(5)<<","<<msg.data.at(6)<<","<<msg.data.at(7)<<","
        <<msg.data.at(8)<<","<<msg.data.at(9)<<","
        <<msg.data.at(10)<<","<<msg.data.at(11)<<"\n";
    }
    // <<msg.data.at(8)<<","<<msg.data.at(9)<<",";
    //  <<msg.data.at(10)<<","<<msg.data.at(11)<<"\n";
    ds_plotdata.close();
}
void drCallback(const std_msgs::Float64MultiArray &msg){
    // Write dr to excel
    std::ofstream dr_plotdata("dr.csv", std::ios::app);
    if (no_of_actuators==2){
        dr_plotdata<<current_goal_set<<","<<msg.data.at(0)<<","<<msg.data.at(1)<<"\n"; //uncomment for 2 joints
    }
    else if (no_of_actuators==3){
        dr_plotdata<<current_goal_set<<","<<msg.data.at(0)<<","<<msg.data.at(1)<<","
        <<msg.data.at(2)<<"\n";//uncomment for 3 joints    
    }   
    
    dr_plotdata.close();
}
void JCallback(const std_msgs::Float32 &msg){
    if(status > 0){
        std::ofstream J_plotdata("modelerror.csv",std::ios::app);
        J_plotdata<<current_goal_set<<","<<msg.data<<"\n";
        J_plotdata.close();
    }
}

void qhatCallback(const std_msgs::Float64MultiArray &msg){
    // Assuming Qhat is a matrix with a fixed number of rows (features) and columns (actuators)
    std::ofstream qhat_plotdata("qhat.csv", std::ios::app);

    // Ensure that the number of elements in the message is as expected for a 8x3 matrix
    if (msg.data.size() == no_of_features * no_of_actuators) {
        for(int row = 0; row < no_of_features; ++row) {
            for(int col = 0; col < no_of_actuators; ++col) {
                // Calculate the index in the flat vector for the current element
                int index = row * no_of_actuators + col;
                qhat_plotdata << msg.data[index];
                // Add a comma after each element except the last one in a row
                if (col < no_of_actuators - 1) {
                    qhat_plotdata << ",";
                }
            }
            // End the line after finishing a row to move to the next row of the matrix
            qhat_plotdata << "\n";
        }
    } else {
        std::cerr << "Error: Received Qhat data size does not match expected size for a 8x3 matrix.\n";
    }
    qhat_plotdata.close();
}

void qhatFeatCallback(const std_msgs::Float64MultiArray &msg){
    // Assuming Qhat is a matrix with a fixed number of rows (features) and columns (actuators)
    std::ofstream qhat_feat_plotdata("qhat_feat.csv", std::ios::app);

    // Ensure that the number of elements in the message is as expected for a 8x3 matrix
    if (msg.data.size() == no_of_actuators*no_of_features) {
        for(int row = 0; row < no_of_actuators; ++row) {
            for(int col = 0; col < no_of_features; ++col) {
                // Calculate the index in the flat vector for the current element
                int index = row * no_of_features + col;
                qhat_feat_plotdata << msg.data[index];
                // Add a comma after each element except the last one in a row
                if (col < no_of_features - 1) {
                    qhat_feat_plotdata << ",";
                }
            }
            // End the line after finishing a row to move to the next row of the matrix
            qhat_feat_plotdata << "\n";
        }
    } else {
        std::cerr << "Error: Received Qhat_feat data size does not match expected size for a 3X8 matrix.\n";
    }
    qhat_feat_plotdata.close();
}

void indModelErrorCallback(const std_msgs::Float64MultiArray &msg) {
    std::ofstream error_data("individual_model_errors.csv", std::ios::app);
    if (error_data.is_open()) {
        for (size_t i = 0; i < msg.data.size(); ++i) {
            error_data << msg.data[i];
            if (i < msg.data.size() - 1) error_data << ", ";
        }
        error_data << "\n";
    }
    error_data.close();
}

// void j1velCallback(const std_msgs::Float64 &msg){
//     std::ofstream j1vel_plotdata("j1vel.csv",std::ios::app);
//     j1vel_plotdata<<msg.data<<"\n";
//     j1vel_plotdata.close();
// }

// void j2velCallback(const std_msgs::Float64 &msg){
//     std::ofstream j2vel_plotdata("j2vel.csv",std::ios::app);
//     j2vel_plotdata<<msg.data<<"\n";
//     j2vel_plotdata.close();
// }

// Uncomment - this block is to save data for 2 joints
// void velCallback(const std_msgs::Float64MultiArray &msg){
//     if(status>0){
//         std::ofstream j1vel_plotdata("j1vel.csv",std::ios::app);
//         j1vel_plotdata<<msg.data.at(0)<<"\n";
//         j1vel_plotdata.close();

//         std::ofstream j2vel_plotdata("j2vel.csv",std::ios::app);
//         j2vel_plotdata<<msg.data.at(1)<<"\n";
//         j2vel_plotdata.close();
//     }
// }

// Unomment - this block is to save data for 3 joints
void velCallback(const std_msgs::Float64MultiArray &msg){
    if(status>0){
        std::ofstream j1vel_plotdata("j1vel.csv",std::ios::app);
        j1vel_plotdata<<current_goal_set<<","<<msg.data.at(0)<<"\n";
        j1vel_plotdata.close();

        std::ofstream j2vel_plotdata("j2vel.csv",std::ios::app);
        j2vel_plotdata<<current_goal_set<<","<<msg.data.at(1)<<"\n";
        j2vel_plotdata.close();
        
        if (no_of_actuators==3){
            std::ofstream j3vel_plotdata("j3vel.csv",std::ios::app);
            j3vel_plotdata<<current_goal_set<<","<<msg.data.at(2)<<"\n";
            j3vel_plotdata.close();
        }
        


    }
}

//  Uncomment the next block for 3f 2j
// void errCallback(const std_msgs::Float64MultiArray &msg){
//     if(status>1){
//         std::ofstream err_plotdata("err.csv", std::ios::app);
//         err_plotdata<<msg.data.at(0)<<","<<msg.data.at(1)<<","
//                     <<msg.data.at(2)<<","<<msg.data.at(3)<<","
//                     <<msg.data.at(4)<<","<<msg.data.at(5)<<"\n";
//         err_plotdata.close();
//     }
// }


// Uncomment the next block for 4f 3j
void errCallback(const std_msgs::Float64MultiArray &msg){
    if(status>1){
        std::ofstream err_plotdata("err.csv", std::ios::app);
        if (no_of_features==8){
            err_plotdata<<current_goal_set<<","<<msg.data.at(0)<<","<<msg.data.at(1)<<","
                    <<msg.data.at(2)<<","<<msg.data.at(3)<<","
                    <<msg.data.at(4)<<","<<msg.data.at(5)<<","
                    <<msg.data.at(6)<<","<<msg.data.at(7)<<"\n";
        }
        if (no_of_features==10){
            err_plotdata<<current_goal_set<<","<<msg.data.at(0)<<","<<msg.data.at(1)<<","
                    <<msg.data.at(2)<<","<<msg.data.at(3)<<","
                    <<msg.data.at(4)<<","<<msg.data.at(5)<<","
                    <<msg.data.at(6)<<","<<msg.data.at(7)<<","
                    <<msg.data.at(8)<<","<<msg.data.at(9)<<"\n";
        }
        else if (no_of_features==12){
            err_plotdata<<current_goal_set<<","<<msg.data.at(0)<<","<<msg.data.at(1)<<","
                    <<msg.data.at(2)<<","<<msg.data.at(3)<<","
                    <<msg.data.at(4)<<","<<msg.data.at(5)<<","
                    <<msg.data.at(6)<<","<<msg.data.at(7)<<","
                    <<msg.data.at(8)<<","<<msg.data.at(9)<<","
                    <<msg.data.at(10)<<","<<msg.data.at(11)<<"\n";
        }
        else if (no_of_features==6){
            err_plotdata<<current_goal_set<<","<<msg.data.at(0)<<","<<msg.data.at(1)<<","
                    <<msg.data.at(2)<<","<<msg.data.at(3)<<","
                    <<msg.data.at(4)<<","<<msg.data.at(5)<<"\n";
        }
        
                    // <<msg.data.at(8)<<","<<msg.data.at(9)<<","
                //    <<msg.data.at(10)<<","<<msg.data.at(11)<<"\n";
        err_plotdata.close();
    }
}

// Uncomment for multiple goals
// void errCallback(const std_msgs::Float64MultiArray &msg) {
//     // Open the file in append mode to add new error data at the end
//     std::ofstream err_plotdata("err.csv", std::ios::app);

//     // Iterate over the error data contained within the msg and write to the file
//     for (const auto& e : msg.data) {
//         err_plotdata << e << ",";
//     }
//     // End each line to separate entries
//     err_plotdata << "\n";
//     err_plotdata.close();
// }

//  Uncomment the next block for 3f 2j
// void cpCallback(const std_msgs::Float64MultiArray &msg){
//         if(status>0){
//             std::ofstream cp_plotdata("cp.csv", std::ios::app);
//             cp_plotdata<<msg.data.at(0)<<","<<msg.data.at(1)<<","<<msg.data.at(2)<<","
//             <<msg.data.at(3)<<","<<msg.data.at(4)<<","<<msg.data.at(5)<<"\n";
//             cp_plotdata.close();
//         }
// }

//Uncomment the next block for 4f 3j
void cpCallback(const std_msgs::Float64MultiArray &msg){
        if(status>0){
            std::ofstream cp_plotdata("cp.csv", std::ios::app);
            if (no_of_features==10){
                cp_plotdata<<current_goal_set<<","<<msg.data.at(0)<<","<<msg.data.at(1)<<","<<msg.data.at(2)<<","
                <<msg.data.at(3)<<","<<msg.data.at(4)<<","<<msg.data.at(5)<<","<<msg.data.at(6)<<","
                <<msg.data.at(7)<<","<<msg.data.at(8)<<","<<msg.data.at(9)<<"\n";
            }
            else if (no_of_features==12){
                cp_plotdata<<current_goal_set<<","<<msg.data.at(0)<<","<<msg.data.at(1)<<","<<msg.data.at(2)<<","
                <<msg.data.at(3)<<","<<msg.data.at(4)<<","<<msg.data.at(5)<<","<<msg.data.at(6)<<","
                <<msg.data.at(7)<<","<<msg.data.at(8)<<","<<msg.data.at(9)<<","
                <<msg.data.at(10)<<","<<msg.data.at(11)<<"\n";
            }
            else if (no_of_features==8){
                cp_plotdata<<current_goal_set<<","<<msg.data.at(0)<<","<<msg.data.at(1)<<","<<msg.data.at(2)<<","
                <<msg.data.at(3)<<","<<msg.data.at(4)<<","<<msg.data.at(5)<<","<<msg.data.at(6)<<","
                <<msg.data.at(7)<<"\n";
            }
            else if (no_of_features==6){
                cp_plotdata<<current_goal_set<<","<<msg.data.at(0)<<","<<msg.data.at(1)<<","<<msg.data.at(2)<<","
            <<msg.data.at(3)<<","<<msg.data.at(4)<<","<<msg.data.at(5)<<"\n";
            }
            
            // <<msg.data.at(8)<<","<<msg.data.at(9)<<","
            // <<msg.data.at(10)<<","<<msg.data.at(11)<<"\n";
            cp_plotdata.close();
        }
}

void jointCallback(const sensor_msgs::JointState::ConstPtr& msg){
    std::ofstream joint_angle_plot("joint_angles.csv", std::ios::app);

    if (msg->position.size() >= 6) { // Ensure there are at least 6 joints
        joint_angle_plot << current_goal_set << ",";
        joint_angle_plot << msg->position[1] << ","; // 2nd joint, index 1
        joint_angle_plot << msg->position[3] << ","; // 4th joint, index 3
        joint_angle_plot << msg->position[5] << "\n"; // 6th joint, index 5
    }
    joint_angle_plot.close();
}


int main(int argc, char **argv){

    // ROS initialization
    ros::init(argc, argv, "servo_control_node");
    ros::NodeHandle n;

    n.getParam("vsbot/shape_control/no_of_features", no_of_features);
    n.getParam("vsbot/shape_control/no_of_actuators", no_of_actuators);

    // std::cout<<"Creating files"<<std::endl;
    // Create files to write data to
    std::ofstream J_plot("modelerror.csv");
    std::ofstream ds_plot("ds.csv");
    std::ofstream dr_plot("dr.csv");
    std::ofstream j1vel_plot("j1vel.csv");
    std::ofstream j2vel_plot("j2vel.csv");
    std::ofstream j3vel_plot("j3vel.csv"); // comment/uncomment on the basis of joint numbers
    std::ofstream err_plot("err.csv");
    std::ofstream cp_plot("cp.csv");
    std::ofstream qhat_plot("qhat.csv");
    std::ofstream qhat_feat_plot("qhat_feat.csv");
    std::ofstream error_data_plot("individual_model_errors.csv");
    // std::ofstream joint_angle_plot("joint_angles.csv");    

    // Add column names to files
    J_plot <<"current_goal"<<","<<"Model Error"<<"\n";
    J_plot.close();

    error_data_plot <<"ind_error1"<<","<<"ind_error2"<<"\n";
    error_data_plot.close();
    // ds_plot <<"ds_x"<<","<<"ds_y"<<"\n";

    // Uncomment the next block for 3f 2j
    // ds_plot <<"cp2 x"<<","<<"cp2 y"<< ","<<"cp3 x"<<","<<"cp3 y"<<","<<"cp4 x"<<","<<"cp4 y"<<"\n";
    // ds_plot.close();
    // dr_plot <<"dr_1"<<","<<"dr_2"<<"\n";
    // dr_plot.close();
    // j1vel_plot <<"Joint 1" <<"\n";
    // j1vel_plot.close();
    // j2vel_plot <<"Joint 2" <<"\n";
    // j2vel_plot.close();
    // j3vel_plot <<"Joint 3" <<"\n";
    // j3vel_plot.close();
    // err_plot <<"Err_cp2_x"<<","<<"Err_cp2_y"<<","<<"Err_cp3_x"<<","<<"Err_cp3_y"
    //          << ","<<"Err_cp4_x"<<","<<"Err_cp4_y"<<"\n";
    // err_plot.close();
    // cp_plot <<"cp2 x"<<","<<"cp2 y"<< ","<<"cp3 x"<<","<<"cp3 y"<<","<<"cp4 x"<<","<<"cp4 y"<<"\n";
    // cp_plot.close();

    // Uncomment next block for 4f 3j
    ds_plot <<"current_goal"<<","<<"cp3 x"<<","<<"cp3 y"<<","<<"cp4 x"<<","<<"cp4 y"<<","
                <<"cp5 x"<<","<<"cp5 y"<<","<<"cp6 x"<<","<<"cp6 y"<<"\n";
    ds_plot.close();
    dr_plot <<"current_goal"<<","<<"dr_1"<<","<<"dr_2"<<","<<"dr_3"<<"\n";
    dr_plot.close();
    j1vel_plot <<"current_goal"<<","<<"Joint 1" <<"\n";
    j1vel_plot.close();
    j2vel_plot <<"current_goal"<<","<<"Joint 2" <<"\n";
    j2vel_plot.close();
    j3vel_plot <<"current_goal"<<","<<"Joint 3" <<"\n";
    j3vel_plot.close();
    err_plot <<"current_goal"<<","<<"Err_cp3_x"<<","<<"Err_cp3_y"
             <<","<<"Err_cp4_x"<<","<<"Err_cp4_y"<<","<<"Err_cp5_x"<<","<<"Err_cp5_y"<<","<<"Err_cp6_x"<<","<<"Err_cp6_y"
             <<"\n";
    err_plot.close();
    cp_plot <<"current_goal"<<","<<"cp3 x"<<","<<"cp3 y"<<","<<"cp4 x"<<","<<"cp4 y"<<","
                <<"cp5 x"<<","<<"cp5 y"<<","<<"cp6 x"<<","<<"cp6 y"<<"\n";
    cp_plot.close();
    // joint_angle_plot<<"current_goal"<<","<<"Joint1"<<","<<"Joint2"<<","<<"Joint3"<<"\n";
    // joint_angle_plot.close();

    qhat_plot << "joint1_jacobian" << "," << "joint2_jacobian" << "," << "joint3_jacobian" << "\n";
    qhat_plot.close();

    qhat_feat_plot << "x1" << "," << "y1" << "," << "x2" << "," << "y2" << "," << "x3" << "," << "y3"
                    <<"," <<"x4"<<","<<"y4"<<"\n";
    qhat_feat_plot.close();


    // Initialize subscribers
    ros::Subscriber ds_sub = n.subscribe("ds_record",1,dsCallback);
    ros::Subscriber dr_sub = n.subscribe("dr_record",1,drCallback);
    ros::Subscriber J_sub = n.subscribe("J_modelerror",1,JCallback);
    ros::Subscriber j_vel_sub = n.subscribe("joint_vel", 1, velCallback);
    ros::Subscriber error_sub = n.subscribe("servoing_error",1,errCallback);
    ros::Subscriber status_sub = n.subscribe("vsbot/status", 1, statusCallback);
    ros::Subscriber cp_sub = n.subscribe("vsbot/control_points", 1, cpCallback);
                                    // This is for storing feature trajectories 
                                    // Subscriber for current goal set
    ros::Subscriber current_goal_set_sub = n.subscribe("current_goal_set_topic", 1, currentGoalSetCallback);
    ros::Subscriber joint_sub = n.subscribe<sensor_msgs::JointState>("joint_states", 1, jointCallback);
    ros::Subscriber Qhat_sub = n.subscribe("Qhat_columns", 1, qhatCallback);
    ros::Subscriber Qhat_feat_sub = n.subscribe("Qhat_rows", 1, qhatFeatCallback);
    ros::Subscriber ind_model_error_sub = n.subscribe("individual_model_error", 1, indModelErrorCallback);




    // Also add subs for Jacobian and Jacobian update

    ros::spin();
    return 0;
}