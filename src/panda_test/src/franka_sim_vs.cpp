#include "ros/ros.h"
#include <rosbag/bag.h>
#include <rosbag/recorder.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <math.h>
#include <iostream>
#include <vector>
#include <numeric>  // For std::inner_product
#include <stdlib.h>

// #include "encoderless_vs/control_points.h"
#include "panda_test/energyFuncMsg.h"
// #include "encoderless_vs/franka_control_points.h"
#include "panda_test/dl_img.h"
#include "panda_test/vel_start.h"

#include "std_msgs/Float32.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"
#include "std_msgs/Int32.h"
#include "std_msgs/Bool.h"



// Status List - 
//  0 - Experiment not started
//  1 - Initial estimation period
//  2 - Visual servoing period
//  3 - Approaching intermediate goals
//  50 - last goal reached
// -1 - Visual servoing completed

// Declare global vector for spline features

int no_of_features; 
int no_of_actuators; 

bool end_flag = false;      // true when servoing is completed. Triggered by user
bool vid_flag = false;
bool start_flag = false;    // true when camera stream is ready

std::vector<float> initial_feature_errors;
std::vector<float> feature_errors;
std::vector<float> final_qhat_initial_estimation;
std::vector<float> init_dS;
std::vector<float> init_dR;

// Initialize a counter for iterations after switching goal sets
int iterations_since_goal_change = 0;
// Define the number of iterations to publish 0 velocities after a goal set change
const int zero_velocity_iterations = 5;


// Sign function for sliding mode controller
int sign(double x){
    if(x<0)
        return -1;
    else if (x>0)
        return 1;
    else
        return 0;
}

void start_flag_callback(const std_msgs::Bool &msg){
    start_flag = msg.data;
}

void end_flag_callback(const std_msgs::Bool &msg){
    end_flag = msg.data;
}

void vid_flag_callback(const std_msgs::Bool &msg){
    vid_flag = msg.data;
}


void print_fvector(std::vector<float> vec){
// function to print std::vector<float>
// this is commonly used for debugging
    for(std::vector<float>::iterator itr=vec.begin(); itr!=vec.end();++itr){
        std::cout<<*itr<<" "<<std::flush;
    }
}


int main(int argc, char **argv){

    // ROS initialization
    ros::init(argc, argv, "shape_servo_control_node");
    ros::NodeHandle n;

    // Initializing ROS publishers
    ros::Publisher j_pub = n.advertise<std_msgs::Float64MultiArray>("joint_vel", 1);
    ros::Publisher ds_pub = n.advertise<std_msgs::Float64MultiArray>("ds_record", 1);
    ros::Publisher dr_pub = n.advertise<std_msgs::Float64MultiArray>("dr_record", 1);
    ros::Publisher J_pub = n.advertise<std_msgs::Float32>("J_modelerror",1);
    ros::Publisher err_pub = n.advertise<std_msgs::Float64MultiArray>("servoing_error", 1);
    ros::Publisher status_pub = n.advertise<std_msgs::Int32>("vsbot/status", 1);
    ros::Publisher cp_pub = n.advertise<std_msgs::Float64MultiArray>("vsbot/control_points", 1);
    // Declaring a publisher for the current goal set
    ros::Publisher current_goal_set_pub = n.advertise<std_msgs::Int32>("current_goal_set_topic", 1);
    ros::Publisher Qhat_pub = n.advertise<std_msgs::Float64MultiArray>("Qhat_columns", 1);
    ros::Publisher Qhat_feat_pub = n.advertise<std_msgs::Float64MultiArray>("Qhat_rows", 1);

    // Initializing ROS subscribers
    ros::Subscriber end_flag_sub = n.subscribe("vsbot/end_flag",1,end_flag_callback);
    ros::Subscriber start_flag_sub = n.subscribe("franka/control_flag", 1, start_flag_callback);
    ros::Subscriber vid_flag_sub = n.subscribe("vsbot/vid_flag", 1, vid_flag_callback);

    
    while(!start_flag){
        ros::Duration(10).sleep();
        ros::spinOnce();
    }

    // waiting for services and camera
    std::cout<<"Sleeping for 10 seconds"<<std::endl;
    ros::Duration(10).sleep();
    
    // Initializing service clients
    ros::service::waitForService("computeEnergyFunc",1000);
    ros::service::waitForService("franka_kp_dl_service", 1000);  

    // Declare Service clients
    ros::ServiceClient kp_client = n.serviceClient<panda_test::dl_img>("franka_kp_dl_service");
    ros::ServiceClient energyClient = n.serviceClient<panda_test::energyFuncMsg>("computeEnergyFunc");
    
    // Initializing status msg
    std_msgs::Int32 status;
    status.data = 0;
    status_pub.publish(status);
    
    // Read number of goals
    int num_goal_sets; 
    n.getParam("dl_controller/num_goal_sets", num_goal_sets);

    // Read in goal features 
    std::vector<std::vector<float>> goal_features;    
    goal_features.resize(num_goal_sets);
    for (int i = 0; i < num_goal_sets; ++i) {
        std::string param_name = "dl_controller/goal_features" + std::to_string(i + 1);
        n.getParam(param_name, goal_features[i]);
    }
    
    // Servoing variables
    int window; // Estimation window size
    n.getParam("vsbot/estimation/window", window);
    int control_window; // Estimation window size
    n.getParam("vsbot/estimation/control_window", control_window);
    float rate; // control & estimation loop rate
    n.getParam("vsbot/estimation/rate", rate);
    float thresh1;
    n.getParam("vsbot/control/thresh1",thresh1);
    float thresh2;
    n.getParam("vsbot/control/thresh2",thresh2);
        float lam;
    n.getParam("vsbot/control/lam",lam);
    std::vector<float> gains1;
    n.getParam("vsbot/control/gains1",gains1);
    std::vector<float> gains2;
    n.getParam("vsbot/control/gains2",gains2);   
    float alpha_gains;
    n.getParam("vsbot/control/alpha_gains",alpha_gains);
    n.getParam("vsbot/shape_control/no_of_features", no_of_features);
    n.getParam("vsbot/shape_control/no_of_actuators", no_of_actuators);
    int debug_mode = 0;
    n.getParam("vsbot/control/debug_mode", debug_mode);

    // Fallback velocity incase Nan velocity generated
    float fallback_vel;
    n.getParam("vsbot/control/fallback_vel", fallback_vel);

    int it = 0;                                     // iterator
    std::vector<float> error (no_of_features,0);    //error vector
    float err = 0.0;                                // error norm
    std_msgs::Float64MultiArray err_msg;            // feature error

    if(debug_mode == 1){
        std::cout << "Initialized Servoing Variables" << std::endl;
    }

    // Estimation variables
    float gamma; // learning rate
    n.getParam("vsbot/estimation/gamma", gamma);    
    // float gamma1; // learning rate during control loop
    // n.getParam("vsbot/estimation/gamma1", gamma1);
    // float gamma2; // learning rate during control loop
    // n.getParam("vsbot/estimation/gamma2", gamma2);
    // float gamma3;
    // n.getParam("vsbot/estimation/gamma3", gamma3);
    // float gamma4;
    // n.getParam("vsbot/estimation/gamma4", gamma4);
    float amplitude;
    n.getParam("vsbot/estimation/amplitude", amplitude);
    float saturation;
    n.getParam("vsbot/control/saturation", saturation);
    std::vector<float> qhat ((no_of_features)*(no_of_actuators), 0);
    n.getParam("vsbot/control/jacobian", qhat);
    std::vector<float> gamma_control_01;
    n.getParam("vsbot/estimation/gamma_control_1", gamma_control_01);
    std::vector<float> gamma_control_02;
    n.getParam("vsbot/estimation/gamma_control_2", gamma_control_02);
    std::cout << "gamma_control_1: " << gamma_control_01[0] << " " << gamma_control_01[1] << " " << gamma_control_01[2] << std::endl;
    std::vector<float> mod_err_thresh;
    n.getParam("vsbot/estimation/mod_err_thresh", mod_err_thresh);

    std::vector<float> ds;          // Current change in key points features
    std::vector<float> dr;          // Current change in joint angles
    std::vector<float> dSinitial;   // Vector list of shape change vectors over sample window
    std::vector<float> dRinitial;   // Vector list of position change vectors over sample window    

    std_msgs::Float64MultiArray j_vel;  // msg to store joint vels
    std_msgs::Float64MultiArray ds_msg; // msg to store current dS window
    std_msgs::Float64MultiArray dr_msg; // msg to store current dR window
    std_msgs::Float64MultiArray control_points; // msg to store control points for current curve

    panda_test::energyFuncMsg msg;

    // Declaring msg for control points service call
    panda_test::dl_img cp_msg;
    cp_msg.request.input = 1;

    float t = 1/rate; // time in seconds, used for integrating angular velocity
    if(debug_mode == 1){
        std::cout <<"Initialized estimation variables" << std::endl;
    }
// --------------------------- Initial Estimation -----------------------------    
// command small displacements around initial position
    ros::Rate r{rate};  // Rate for control loop
    if(debug_mode == 1){
        std::cout << "Ready to command small displacements" <<std::endl; 
    }
    
    // Obtain initial robot state
    std::vector<float> cur_features(no_of_features, 0);
    kp_client.call(cp_msg);
    control_points.data.clear();
    for(int i = 0; i<no_of_features; i++){
        cur_features[i] = cp_msg.response.kp.data.at(i);
        control_points.data.push_back(cur_features[i]);
    }

    // set old_features to first set of features received
    std::vector<float> old_features = cur_features;

    // Change status msg to initial estimation
    status.data = 1;

    // parameter for generating joint velocities
    float param = 0.3; // starting value for joint velocity

    std::vector<float> initial_feature_errors(no_of_features, 0.0);

    if (debug_mode == 2){
        std::cout << "The total window size before any iteration in estimation loop: " << dSinitial.size() << std::endl;
        std::cout << "The n window size before any iteration in estimation loop: " << ds.size() << std::endl;
    }
    
    // Collecting data for estimation window
    while (it < window){

        // Publish sin vel to both joints
        float j1_vel = amplitude*sin(param);
        float j2_vel = amplitude*cos(param); 
        float j3_vel;
        if (no_of_actuators == 3){
            j3_vel = amplitude*(cos(param)+sin(param)); // comment out when 3rd joint not in use
        }
        
        param = param + 0.1;
        
        j_vel.data.clear();
        j_vel.data.push_back(j1_vel);
        j_vel.data.push_back(j2_vel);

        if (no_of_actuators == 3) {
            j_vel.data.push_back(j3_vel); // comment out when 3rd joint not in use
        }
        j_pub.publish(j_vel);

        // Obtain current robot state
        kp_client.call(cp_msg);
        control_points.data.clear();
        
        for(int i = 0; i<no_of_features; i++){
            cur_features[i] = cp_msg.response.kp.data.at(i);
            control_points.data.push_back(cur_features[i]);
        }
        cp_pub.publish(control_points);

        // Compute change in state
        // shape features
        ds.clear();
        for(int i = 0; i<old_features.size(); i++){
            ds.push_back((cur_features[i] - old_features[i]));
        }

        //  Joint angle (change in robot configuration)
        dr.clear();
        dr.push_back((j1_vel*t));
        dr.push_back((j2_vel*t));
        if (no_of_actuators==3){
            dr.push_back((j3_vel*t)); // comment out when 3rd joint not in use
        }

        // Update dSinitial and dRinitial
        for(int i = 0; i < no_of_features; i++){
            dSinitial.push_back(ds[i]);
        }

        for(int i = 0; i < no_of_actuators;i++){
            dRinitial.push_back(dr[i]);
        }

        init_dS = dSinitial;

        init_dR = dRinitial;

        for (int i = 0; i < no_of_features; i++) {
            initial_feature_errors[i] = std::abs(cur_features[i] - old_features[i]);
        }        

        if (debug_mode == 2){
        std::cout << "The total window size after each iteration in estimation loop: " << dSinitial.size() << std::endl;
        std::cout << "The n window size after each iteration in estimation loop: " << ds.size() << std::endl;
        }
        
        // Update state variables
        old_features = cur_features;

        // Publish ds, dr vectors to store
            // Convert to Float64multiarray
        ds_msg.data.clear();
        for(int i = 0; i < no_of_features; i++){
            ds_msg.data.push_back(ds[i]);
        }

        dr_msg.data.clear();
        for(int i = 0; i < no_of_actuators; i++){
            dr_msg.data.push_back(dr[i]);
        }
        
        // publish
        ds_pub.publish(ds_msg);
        dr_pub.publish(dr_msg);

        ds.clear();
        dr.clear();
     
        // publish status msg
        status_pub.publish(status);

        //Increase iterator 
        it++;

        // Refresh subscriber callbacks
        ros::spinOnce();
        r.sleep();     
    }
    
    
    // Commanding 0 velocity to robot 
    j_vel.data.clear();
    j_vel.data.push_back(0.0);
    j_vel.data.push_back(0.0);
    if (no_of_actuators==3){
        j_vel.data.push_back(0.0); //comment/uncomment depending on number of joints
    }

    j_pub.publish(j_vel);

    if(debug_mode == 1){
        std::cout<<"Initial Movements Complete"<<std::endl;
    }

    // Declare ROS Msg Arrays
    std_msgs::Float32MultiArray dSmsg;
    dSmsg.layout.dim.push_back(std_msgs::MultiArrayDimension());
    dSmsg.layout.dim[0].label = "dS_elements";
    dSmsg.layout.dim[0].size = dSinitial.size();
    dSmsg.layout.dim[0].stride = 1;
    dSmsg.data.clear();
    
    std_msgs::Float32MultiArray dRmsg;
    dRmsg.layout.dim.push_back(std_msgs::MultiArrayDimension());
    dRmsg.layout.dim[0].label = "dR_elements";
    dRmsg.layout.dim[0].size = dRinitial.size();
    dRmsg.layout.dim[0].stride = 1;
    dRmsg.data.clear();

    std_msgs::Float32MultiArray qhatmsg;
    qhatmsg.layout.dim.push_back(std_msgs::MultiArrayDimension());
    qhatmsg.layout.dim[0].label = "qhat_elements";
    qhatmsg.layout.dim[0].size = qhat.size();
    qhatmsg.layout.dim[0].stride = 1;
    qhatmsg.data.clear();

    // Push data to ROS Msg
    for(std::vector<float>::iterator itr = dSinitial.begin(); itr != dSinitial.end(); ++itr){
        // std::cout <<*itr<<std::endl;
        dSmsg.data.push_back(*itr);
    }

    for(std::vector<float>::iterator itr = dRinitial.begin(); itr != dRinitial.end(); ++itr){
        // std::cout <<*itr<<std::endl;
        dRmsg.data.push_back(*itr);
    }

    for(std::vector<float>::iterator itr = qhat.begin(); itr != qhat.end(); ++itr){
        // std::cout <<*itr<<std::endl;
        qhatmsg.data.push_back(*itr);
    }  

    if(debug_mode == 1){
        std::cout <<"Pushed initial data to ROS msgs"<<std::endl;
    }

    std_msgs::Float32MultiArray initial_feature_errors_msg;
    initial_feature_errors_msg.data = initial_feature_errors;

    // Compute Jacobian
    it = 0;   
    while(it < window){
        int cur_win_size = dSinitial.size()/no_of_features;
        // Service request data
        // msg.request.gamma_general = gamma;
        msg.request.gamma_first_actuator = gamma;
        msg.request.gamma_second_actuator = gamma;
        msg.request.gamma_third_actuator = gamma;
        msg.request.it = it;
        msg.request.dS = dSmsg;
        msg.request.dR = dRmsg;
        msg.request.qhat = qhatmsg;
        msg.request.feature_error_magnitude = 1.0;
        msg.request.feature_errors = initial_feature_errors_msg;
        msg.request.data_size = window;

        // call compute energy functional
        energyClient.call(msg);
        // if(debug_mode==1){
        //     std::cout << "Is the energyClient called: " << std::endl;
        // }

        // Populating service response
        std::vector<float> qhatdot = msg.response.qhat_dot.data;

        //  Jacobian update
        // std::cout<<"size of qhat:"<<qhat.size()<<std::endl;
        for(int i = 0; i<qhat.size(); i++){
            qhat[i] = qhat[i] + qhatdot[i]; // Updating each element of Jacobian
        }
        // std::cout<<"Updated Jacobian vector:";

        // Push updated Jacobian vector to ROS Msg
        qhatmsg.data.clear();
        for(std::vector<float>::iterator itr = qhat.begin(); itr != qhat.end(); ++itr){
            qhatmsg.data.push_back(*itr);
        }

        // Save the final Jacobian from the initial estimation loop
        // final_qhat_initial_estimation = qhat;
    
        // Print the contents of qhat
   
        // Publish J value to store
        std_msgs::Float32 J;
        J.data = msg.response.J;
        J_pub.publish(J);
        
        // Increase iterator
        it++;
    }
    if(debug_mode == 1){
        std::cout <<"Initial Estimation Completed" << std::endl;
    }

    if(debug_mode ==1){
        std::cout<<"initial Jacobian estimation complete"<<std::endl;
    }

// ----------------------------- Start Servoing ---------------------------------- 
    // err = thresh; // set error norm to threshold to start control loop
    if(debug_mode == 1){
        std::cout<<"Entering control loop"<<std::endl;
    }
    // Initialize the first goal set
    std::vector<float> goal = goal_features[0]; // Start with the first goal set
    int current_goal_set = 0; // Index of the current goal feature set

    // Switching to control loop rate
    t = 1/rate;
    ros::Rate control_r{rate};    

    // Publish status
    status.data = 2;
    status_pub.publish(status);

    // Refresh subscribers
    ros::spinOnce();

    // ----------------------- Control Loop for Servoing -----------------------------    
    std_msgs::Int32 current_goal_set_msg; // Message for publishing current goal set index

    while(ros::ok() && !end_flag && current_goal_set < num_goal_sets){    // convergence condition

        bool fallback_used = false;

        // error norm "err" is always positive
        // compute current error & norm
        for(int i = 0; i < no_of_features; i++){
            error[i] = cur_features[i] - goal[i];
        }
        float err_acc = 0; // accumulator vairable for computing error norm
        for(int i=0; i<no_of_features; i++){
            err_acc += error[i]*error[i];
        }
        err = sqrt(err_acc);
        err_acc = 0; // Reset error accumulator

        std::vector<float> current_gains;   
        std::vector<float> gamma_control; 
        float cur_sat;


        // current_gains = gains1;
        if ((err <= mod_err_thresh[1])) {
                        current_gains = gains1;
            gamma_control = gamma_control_01;
            }
        else {
            gamma_control = gamma_control_02;
            current_gains = gains2; // Use the second set of gains for the last goal
            cur_sat = saturation[0];
        }
        // current_gains = gains1;
        if(debug_mode == 1) {
            // std::cout << "Current goal set: " << current_goal_set << std::endl;
            // std::cout << "Number of goal sets: " << num_goal_sets << std::endl;
            // std::cout << "Current Threshold in use: " << ((current_goal_set < num_goal_sets - 1) ? thresh1 : thresh2) << std::endl;
            std::cout << "Current Error: " << err << std::endl;
        }
        // std::cout << "gamma: " << gamma_control[0] << " " << gamma_control[1] << " " << gamma_control[2] << std::endl;

        // Convert gains to Eigen vector
        Eigen::VectorXf K(no_of_features);
        for(int i=0; i<no_of_features; i++){
            K[i] = current_gains[i];
        }
        // Generate velocity
        // Convert qhat vector into matrix format
        Eigen::MatrixXf Qhat(no_of_features,no_of_actuators);
        int row_count = 0;
        int itr = 0;
        while(row_count<no_of_features){
            for (int j = 0; j < no_of_actuators; j++){
                Qhat(row_count, j) = qhat[itr+j];
            }
            row_count = row_count + 1;
            itr = itr + no_of_actuators;
        }
                std_msgs::Float64MultiArray Qhat_msg;
        Qhat_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        Qhat_msg.layout.dim[0].label = "Qhat_column_elements";
        Qhat_msg.layout.dim[0].size = Qhat.rows() * Qhat.cols(); // Assuming Qhat is a dense matrix
        Qhat_msg.layout.dim[0].stride = 1;
        // Flatten Qhat into Qhat_msg.data
        for (int row = 0; row < Qhat.rows(); ++row) {
            for (int col = 0; col < Qhat.cols(); ++col) {
                Qhat_msg.data.push_back(Qhat(row, col));
            }
        }
        // Publishing the Qhat matrix
        Qhat_pub.publish(Qhat_msg);
        Eigen::MatrixXf Qhat_T = Qhat.transpose();
        if(debug_mode == 2){
            // CHECK CONDITION NUMBER OF JACOBIAN
            Eigen::JacobiSVD<Eigen::MatrixXf> svd(Qhat, Eigen::ComputeFullU | Eigen::ComputeFullV);
            float tolerance = 1e-4; // Threshold for considering singular values as zero
            bool is_singular = svd.singularValues().minCoeff() < tolerance;
            // Optional: Print the condition number for diagnostics
            float cond_number = svd.singularValues().maxCoeff() / svd.singularValues().minCoeff();
        } 
        // Convert error std::vector to Eigen::vector
        // for matrix computations
        Eigen::VectorXf error_vec(no_of_features);
        for(int i=0; i<no_of_features; i++){
            error_vec(i) = error[i];
        }
        Eigen::VectorXf unsaturated_error_vec = error_vec;
        // joint velocity Eigen::vector
        Eigen::VectorXf joint_vel;
        if (no_of_actuators==3){
            Eigen::Vector3f joint_vel;  
        }
        else if (no_of_actuators==2) {
            Eigen::Vector2f joint_vel;
        }       
        // Closed form solution for linearly independent columns
        // A_inv = (A.transpose()*A).inverse() * A.transpose()
        // Applying SVD based pseudo inverse
        // Eigen::JacobiSVD<Eigen::MatrixXf> svd(Qhat, Eigen::ComputeThinU | Eigen::ComputeThinV);
        // float tolerance = 1e-6;
        // Eigen::VectorXf singular_values = svd.singularValues();
        // for (int i = 0; i < singular_values.size(); ++i) {
        //     if (singular_values(i) < tolerance) singular_values(i) = 0.0;
        //     else singular_values(i) = 1.0 / singular_values(i);
        // }
        // Damped Pseudo-Inverse (Tikhonov regularization)
        float lambda = 1e-4;
        // Eigen::MatrixXf Qhat_inv = (Qhat.transpose() * Qhat + lambda * Eigen::MatrixXf::Identity(Qhat.cols(), Qhat.cols())).inverse() * Qhat.transpose();


        // Calls svd internally - supposedly more efficient
        // Eigen::MatrixXf Qhat_inv = Qhat.completeOrthogonalDecomposition().pseudoInverse();
        // Eigen::MatrixXf Qhat_inv = svd.matrixV() * singular_values.asDiagonal() * svd.matrixU().transpose();
        Eigen::MatrixXf Qhat_inv = (Qhat.transpose()*Qhat).inverse() * Qhat.transpose();
        std::cout << "Is Qhat Inverse the issue for NaN" << Qhat_inv << std::endl; 

        std::cout << "Current saturation" << cur_sat << std::endl; 
        // Saturating the error
        for(int i=0; i<no_of_features; i++){
            if(abs(error_vec(i)) > cur_sat){
                    if (abs(error_vec(i)) < 1e-6) error_vec(i) = 0.0;
                    else error_vec(i) = (error_vec(i)/abs(error_vec(i))) * cur_sat;
                }
            }
                    // std::cout << "Erroc Vector after saturation" << error_vec << std::endl; 
        std_msgs::Float64MultiArray Qhat_feat_msg;
        Qhat_feat_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        Qhat_feat_msg.layout.dim[0].label = "Qhat_row_elements";
        Qhat_feat_msg.layout.dim[0].size = Qhat_inv.rows() * Qhat_inv.cols(); // Assuming Qhat is a dense matrix
        Qhat_feat_msg.layout.dim[0].stride = 1;
        // Flatten Qhat into Qhat_msg.data
        for (int row = 0; row < Qhat_inv.rows(); ++row) {
            for (int col = 0; col < Qhat_inv.cols(); ++col) {
                Qhat_feat_msg.data.push_back(Qhat_inv(row, col));
            }
        }
        // Publishing the Qhat matrix
        Qhat_feat_pub.publish(Qhat_feat_msg);
        // Compute the error magnitude for adaptive gain
        float error_magnitude = unsaturated_error_vec.norm();
        feature_errors.clear();
        for (int i = 0; i < no_of_features; i++) {
            float feature_error = abs(unsaturated_error_vec(i));       
            feature_errors.push_back(feature_error);
        } 
        std_msgs::Float32MultiArray feature_errors_msg;
        feature_errors_msg.data = feature_errors;
        joint_vel = (Qhat_inv)*(Eigen::MatrixXf(K.asDiagonal())*error_vec);
        if (debug_mode == 1){
            std::cout << "Current Joint_vel: " << joint_vel << std::endl;
        }
        // Publish velocity to robot
        // Check for NaNs in the computed velocities
        // if (!joint_vel.allFinite()) {
        //     std::cerr << "Warning: NaN detected in joint velocities. Applying minimal velocities." << std::endl;
        // }
        j_vel.data.clear();

        for (int i =0; i < no_of_actuators; ++i) {
            float vel_i = joint_vel[i];
            if (!std::isfinite(vel_i)) {
                std::cerr << "Warning: NaN or Inf detected in joint " << i 
                  << ". Applying fallback velocity to this joint." << std::endl;

                vel_i = fallback_vel;  // Replace with your fallback value for that joint
                fallback_used = true;
            }

            j_vel.data.push_back(vel_i);
        }

        // j_vel.data.push_back(joint_vel[0]);
        // j_vel.data.push_back(joint_vel[1]);
        // if (err <= 150) {
        //     j_vel.data.push_back(joint_vel[2]);
        // }
        // else {
        //     j_vel.data.push_back(joint_vel[2]*0.5);
        // }
        // if (no_of_actuators==3) {
        //     j_vel.data.push_back(joint_vel[2]); // Only add j3_vel if no_of_actuators is more than 2
        // }
        j_pub.publish(j_vel);  
        
        // Get current state of robot
        control_points.data.clear();
        kp_client.call(cp_msg);
        for(int i = 0; i<no_of_features; i++){
            cur_features[i] = cp_msg.response.kp.data.at(i);
            control_points.data.push_back(cur_features[i]);
        }
        // Compute change in state
        ds.clear();
        for(int i=0; i<no_of_features;i++){
            ds.push_back((cur_features[i]-old_features[i]));
        }
        dr.clear();
        for(int i=0; i<no_of_actuators;i++){
            dr.push_back(joint_vel[i]*t);
        }  
        float err = sqrt(std::inner_product(error.begin(), error.end(), error.begin(), 0.0));
        
        // Evaluate errors for all remaining goals
        bool found_closer_goal = false;
        int next_goal_set = current_goal_set;
        for (int i = current_goal_set; i < num_goal_sets; ++i) {
            std::vector<float> temp_goal = goal_features[i];
            float temp_err = 0.0;
            for (int j = 0; j < no_of_features; j++) {
                float temp_error = cur_features[j] - temp_goal[j];
                temp_err += temp_error * temp_error;
            }
            temp_err = sqrt(temp_err);

            // float current_thresh = (i == num_goal_sets - 1) ? thresh2 : thresh1;
            float current_thresh = thresh1;            


            if (temp_err < current_thresh) {
                found_closer_goal = true;
                next_goal_set = i;
                break;
            }
        }
        // if (found_closer_goal && next_goal_set != current_goal_set) {
            
        //     current_goal_set = next_goal_set;
        //     goal = goal_features[current_goal_set]; // Update the goal to the closest set

        //     // Clear ds and dr windows
        //     dSinitial.clear();
        //     dRinitial.clear();
        //     // dSinitial = init_dS;
        //     // dRinitial = init_dR;
        //     std::cout << "Moving to closer goal set " << current_goal_set << std::endl;
        // }
        // Check if we should move to a different goal
                if (err < ((current_goal_set < num_goal_sets - 1) ? thresh1 : thresh2)) {
            // If the current goal is reached
            std::cout << "Goal " << current_goal_set << " reached. Moving to next goal." << std::endl;
            if (current_goal_set < num_goal_sets - 1) {
                ++current_goal_set; // Move to the next set of goal features
                goal = goal_features[current_goal_set]; // Update the goal to the next set

                // Set status to 3 for intermediate goal
                status.data = 2+current_goal_set;
                status_pub.publish(status);

                std::cout << "Approaching intermediate goal, setting status = per goal_status." << std::endl;

                // Clear ds and dr windows
                dSinitial.clear();
                dRinitial.clear();
                // dSinitial = init_dS;
                // dRinitial = init_dR;
            } else {
                // All goals reached
                std::cout << "All goals reached" << std::endl;
                // break;
            }
        }       
        else{                 
            // Update sample window
            int data_size = dSinitial.size()/no_of_features;
            if(debug_mode==1){
                    std::cout << "N Size before condition : " << data_size << std::endl;
                    std::cout << "n Size before condition : " << ds.size() << std::endl;
            }
            if (data_size < control_window) {
                // if(debug_mode==2){
                //     std::cout << "Current Window Size for small n: " << data_size << std::endl;
                // }
                for(int i=0; i<no_of_features;i++){
                    dSinitial.push_back(ds[i]);
                }
                for(int i=0; i<no_of_actuators;i++){
                    dRinitial.push_back(dr[i]);
                }
            }
            else {
                // Discard oldest and push latest sample
                for(int i=0; i<no_of_features;i++){
                    dSinitial[i] = ds[i];
                }
                for(int i=0; i<no_of_actuators;i++){
                    dRinitial[i] = dr[i];
                }
                std::rotate(dSinitial.begin(), dSinitial.begin()+no_of_features, dSinitial.end());
                std::rotate(dRinitial.begin(), dRinitial.begin()+no_of_actuators, dRinitial.end());              
            }          
            
            int cur_win_size = dSinitial.size()/no_of_features;
            // Compute Jacobian update with new sampling window
            // converting vectors to ros msg for service
            dSmsg.data.clear();
            for(std::vector<float>::iterator itr = dSinitial.begin(); itr != dSinitial.end(); ++itr){
                dSmsg.data.push_back(*itr);
            }
            dRmsg.data.clear();
            for(std::vector<float>::iterator itr = dRinitial.begin(); itr != dRinitial.end(); ++itr){
                dRmsg.data.push_back(*itr);
            }
            qhatmsg.data.clear();
            for(std::vector<float>::iterator itr = qhat.begin(); itr != qhat.end(); ++itr){
                qhatmsg.data.push_back(*itr);
            }
            // std::cout << "gamma: " << gamma_control[0] << " " << gamma_control[1] << " " << gamma_control[2] << std::endl;

            msg.request.gamma_first_actuator = gamma_control[0];
            msg.request.gamma_second_actuator = gamma_control[1];
            msg.request.gamma_third_actuator = gamma_control[2];
            msg.request.it = cur_win_size - 1;
            msg.request.dS = dSmsg;
            msg.request.dR = dRmsg;
            msg.request.qhat = qhatmsg;
            msg.request.feature_error_magnitude = error_magnitude; 
            msg.request.feature_errors = feature_errors_msg; 
            msg.request.data_size = cur_win_size;

            energyClient.call(msg);
            if (!energyClient.call(msg)) {
                ROS_ERROR("Failed to call service computeEnergyFunc");
                break;  
            }
            else {
                std::cout << "The energyClient called: " << std::endl;
            }
            std::vector<float> qhatdot = msg.response.qhat_dot.data;
            // Update Jacobian
            for(int i = 0; i<qhat.size(); i++){
                qhat[i] = qhat[i] + qhatdot[i]; // Updating each element of Jacobian
            }
            // Push updated Jacobian vector to ROS Msg
            qhatmsg.data.clear();
            for(std::vector<float>::iterator itr = qhat.begin(); itr != qhat.end(); ++itr){
                qhatmsg.data.push_back(*itr);
            }            

            // Update state variables
            old_features = cur_features;
            // Publish ds, dr, J, & error vectors to store
            // Convrt to Float64multiarray
            ds_msg.data.clear();
            for(int i=0; i<no_of_features; i++){
                // std::cout << "Element " << i << " value: " << ds[i] << std::endl;
                ds_msg.data.push_back(ds[i]);
            }
            dr_msg.data.clear();
            for(int i=0; i<no_of_actuators; i++){
                dr_msg.data.push_back(dr[i]);
            }
            ds_pub.publish(ds_msg);
            dr_pub.publish(dr_msg);      
        }

        err_msg.data.clear();
        for(int i = 0; i<no_of_features;i++){
            err_msg.data.push_back(error[i]);
        }

        if (vid_flag && !end_flag) {
            status.data = 50;
            status_pub.publish(status); 
        }

        // publish
        std_msgs::Float32 J;
        J.data = msg.response.J;

        J_pub.publish(J);

        // Publish control points
        cp_pub.publish(control_points);
        
        err_pub.publish(err_msg);        
        
        // Publish status msg
        status_pub.publish(status);

        // // Publish the updated current goal set index
        current_goal_set_msg.data = current_goal_set;
        current_goal_set_pub.publish(current_goal_set_msg);

        // Refresh subscriber callbacks
        ros::spinOnce();
        control_r.sleep();
    }

    // Commanding 0 velocity to robot 
    j_vel.data.clear();
    j_vel.data.push_back(0.0);
    j_vel.data.push_back(0.0);
    if (no_of_actuators==3){
        j_vel.data.push_back(0.0); 
    }

    j_pub.publish(j_vel);

    std::cout<<"Servoing Complete"<<std::endl;
    status.data = -1;
    status_pub.publish(status);
    
    // Shutdown
    // Status flag will shutdown record node which is tied to all other nodes
    // This is done so all the recorded files can be closed and saved safely before
    // the nodes shut down

    ros::spin();
    return 0;
}

