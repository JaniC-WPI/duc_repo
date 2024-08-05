#include "ros/ros.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <math.h>
#include <iostream>
#include <vector>
#include <numeric>  // For std::inner_product

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
// -1 - Visual servoing completed

// Declare global vector for spline features

int no_of_features; // = 4; // 3 control points in a plane, 
    // ignoring 1st control pt as it doesn't change much and can be discarded

int no_of_actuators; 

bool end_flag = false;      // true when servoing is completed. Triggered by user
bool start_flag = false;    // true when camera stream is ready

std::vector<float> initial_feature_errors;
std::vector<float> feature_errors;
std::vector<float> final_qhat_initial_estimation;

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
    // std::cout << "Initialized Publishers" <<std::endl;

    // Initializing ROS subscribers
    ros::Subscriber end_flag_sub = n.subscribe("vsbot/end_flag",1,end_flag_callback);
    ros::Subscriber start_flag_sub = n.subscribe("franka/control_flag", 1, start_flag_callback);
    std::cout<<start_flag<<std::endl;
    while(!start_flag){
        ros::Duration(10).sleep();
        std::cout<<"Waiting for camera"<<std::endl;
        ros::spinOnce();
    }

    // waiting for services and camera
    std::cout<<"Sleeping for 10 seconds"<<std::endl;
    ros::Duration(10).sleep();
    
    // Initializing service clients
    ros::service::waitForService("computeEnergyFunc",1000);
    // std::cout<<"Compute Energy Func is waited"<<std::endl;
    // ros::service::waitForService("franka_control_service", 1000);       
                                // this service generates control points
    ros::service::waitForService("franka_kp_dl_service", 1000);  
    // std::cout<<"franka_kp_service is waited"<<std::endl;
                                // this service genarates key_points for dream

    // ros::service::waitForService("binary_image_output", 1000);

    ros::ServiceClient energyClient = n.serviceClient<panda_test::energyFuncMsg>("computeEnergyFunc");
    // std::cout<<"Compute Energy func is getting called ?"<<std::endl;
    // ros::ServiceClient cp_client = n.serviceClient<encoderless_vs::franka_control_points>("franka_control_service");

    // Added client for dream kp generation
    ros::ServiceClient kp_client = n.serviceClient<panda_test::dl_img>("franka_kp_dl_service");
    // std::cout<<"franka kp service is getting called ?"<<std::endl;

    // Initializing status msg
    std_msgs::Int32 status;
    status.data = 0;
    status_pub.publish(status);

    // Load multiple goal features from the parameter server
    std::vector<std::vector<float>> goal_features; 
    int num_goal_sets; 
    n.getParam("dl_controller/num_goal_sets", num_goal_sets);
    goal_features.resize(num_goal_sets);
    // std::cout<<"goal feature size"<<goal_features.size()<<std::endl;
    for (int i = 0; i < num_goal_sets; ++i) {
        std::string param_name = "dl_controller/goal_features" + std::to_string(i + 1);
        n.getParam(param_name, goal_features[i]);
    }
    // std::cout<<"goal feature size"<<goal_features.size()<<std::endl;
    // print_fvector(goal_features);
    // Servoing variables
    int window; // Estimation window size
    n.getParam("vsbot/estimation/window", window);

    float rate; // control & estimation loop rate
    n.getParam("vsbot/estimation/rate", rate);

    // float control_rate;
    // n.getParam("vsbot/control/rate", control_rate);

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

    float reg_lambda;
    n.getParam("vsbot/control/reg_lambda",reg_lambda);

    float p_lam;
    n.getParam("vsbot/control/p_lam",p_lam);

    float gain_sm;
    n.getParam("vsbot/control/gain_sm", gain_sm);

    n.getParam("vsbot/shape_control/no_of_features", no_of_features);
    n.getParam("vsbot/shape_control/no_of_actuators", no_of_actuators);

    // std::vector<float> goal (no_of_features,0);
    // n.getParam("dl_controller/goal_features", goal);
    // // print_fvector(goal);

    int it = 0;                                     // iterator
    std::vector<float> error (no_of_features,0);    //error vector
    float err = 0.0;                                // error norm
    std_msgs::Float64MultiArray err_msg;            // feature error
    // std::cout << "Initialized Servoing Variables" << std::endl;

    // Estimation variables
    float gamma; // learning rate
    n.getParam("vsbot/estimation/gamma", gamma);
    
    float gamma1; // learning rate during control loop
    n.getParam("vsbot/estimation/gamma1", gamma1);
    float gamma2; // learning rate during control loop
    n.getParam("vsbot/estimation/gamma2", gamma2);

    float gamma3;
    n.getParam("vsbot/estimation/gamma3", gamma3);

    float beta; // threshold for selective Jacobian update
    n.getParam("vsbot/shape_control/beta", beta);

    float amplitude;
    n.getParam("vsbot/estimation/amplitude", amplitude);

    float saturation;
    n.getParam("vsbot/control/saturation", saturation);

    std::vector<float> qhat ((no_of_features)*(no_of_actuators), 0);
    n.getParam("vsbot/control/jacobian", qhat);

    std::vector<float> ds; // change in key points features
    std::vector<float> dr; // change in joint angles

    // Changed this to 8 elements since only using 4 features now

    std::vector<float> dSinitial; // Vector list of shape change vectors
    std::vector<float> dRinitial; // Vector list of position change vectors

    std_msgs::Float64MultiArray j_vel;  // msg to store joint vels
    std_msgs::Float64MultiArray ds_msg; // msg to store current dS window
    std_msgs::Float64MultiArray dr_msg; // msg to store current dR window
    // std_msgs::Float64 dth_msg; // msg to store current dTh

    std_msgs::Float64MultiArray control_points; // msg to store control points for current curve

    // Declaring msg for control points service call
    // encoderless_vs::franka_control_points cp_msg;
    panda_test::dl_img cp_msg;
    cp_msg.request.input = 1;

    float t = 1/rate; // time in seconds, used for integrating angular velocity
    // std::cout <<"Initialized estimation variables" << std::endl;

    // uncomment next block for 3D motions

// --------------------------- Initial Estimation -----------------------------    


// command small displacements around initial position
    ros::Rate r{rate};  // Rate for control loop
    std::cout << "Ready to command small displacements" <<std::endl; 
    
    // Obtain initial robot state
    std::vector<float> cur_features(no_of_features, 0);
    // cp_client.call(cp_msg);
    kp_client.call(cp_msg);
    control_points.data.clear();
    for(int i = 0; i<no_of_features; i++){
        // cur_features[i] = cp_msg.response.cp.data.at(i);
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
    
    // Collecting data for estimation window
    while (it < window){

        // Publish sin vel to both joints
        float j1_vel = amplitude*sin(param);
        float j2_vel = amplitude*cos(param); 
        float j3_vel;
        if (no_of_actuators == 3){
            j3_vel = amplitude*(cos(param)+sin(param)); // comment out when 3rd joint not in use
        }

        // float j1_vel = amplitude*sin(param);
        // float j2_vel = amplitude*cos(param); 
        
        param = param + 0.1;
        
        // Adding noise to sinusoidal velocity
        // j1_vel.data += (2*((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - 0.5));
        // j2_vel.data += (2*((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - 0.5));
        
        j_vel.data.clear();
        j_vel.data.push_back(j1_vel);
        j_vel.data.push_back(j2_vel);

        if (no_of_actuators == 3) {
            j_vel.data.push_back(j3_vel); // comment out when 3rd joint not in use
        }
        j_pub.publish(j_vel);

        // Obtain current robot state
        // cp_client.call(cp_msg);
        kp_client.call(cp_msg);
        control_points.data.clear();
        
        // std::cout<<"# Features: " << no_of_features <<std::endl;

        for(int i = 0; i<no_of_features; i++){
            // cur_features[i] = cp_msg.response.cp.data.at(i);
            cur_features[i] = cp_msg.response.kp.data.at(i);
            control_points.data.push_back(cur_features[i]);
        }
        cp_pub.publish(control_points);
        // print_fvector(cur_features);

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
        // dRinitial.push_back(dr[0]);
        // dRinitial.push_back(dr[1]);
        // dRinitial.push_back(dr[2]); // Comment for 3rd joint

         for (int i = 0; i < no_of_features; i++) {
            initial_feature_errors[i] = std::abs(cur_features[i] - old_features[i]);
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

        // Publish control points
        // cp_pub.publish(control_points);

        // publish status msg
        status_pub.publish(status);

        //Increase iterator 
        // std::cout <<"iterator:" << it <<std::endl;
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

    std::cout<<"Initial Movements Complete"<<std::endl;

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
    // std::cout << "Declared ROS msg arrays" <<std::endl;

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

    std::cout <<"Pushed initial data to ROS msgs"<<std::endl;

    std_msgs::Float32MultiArray initial_feature_errors_msg;
    initial_feature_errors_msg.data = initial_feature_errors;

    // Compute Jacobian
    it = 0;
    panda_test::energyFuncMsg msg;
    while(it < window){
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

        // call compute energy functional
        energyClient.call(msg);

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
        final_qhat_initial_estimation = qhat;
    
        // Print the contents of qhat
        std::cout << "final qhat in initial estimation: ";
            for (const auto& val : qhat) {
                std::cout << val << " ";
            }
        std::cout << std::endl;

        // Publish J value to store
        std_msgs::Float32 J;
        J.data = msg.response.J;
        J_pub.publish(J);
        
        // Increase iterator
        it++;
    }
    std::cout <<"Initial Estimation Completed" << std::endl;

// ----------------------------- Start Servoing ---------------------------------- 
    // err = thresh; // set error norm to threshold to start control loop
    std::cout<<"Entering control loop"<<std::endl;
    // Initialize the first goal set
    std::vector<float> goal = goal_features[0]; // Start with the first goal set
    // std::cout<<"print goals"<<std::endl;
    // print_fvector(goal);
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
        // error norm "err" is always positive
        // std::vector<float> cur_goal = goal_features[current_goal_set];
        // compute current error & norm
        // print_fvector(cur_features);
        // print_fvector(goal);
        for(int i = 0; i < no_of_features; i++){
            error[i] = cur_features[i] - goal[i];
        }
    
        float err_acc = 0; // accumulator vairable for computing error norm
        for(int i=0; i<no_of_features; i++){
            err_acc += error[i]*error[i];
        }
        err = sqrt(err_acc);
        err_acc = 0; // Reset error accumulator
        // std::cout<<" norm:"<<err<<std::endl;

        std::vector<float> current_gains;
        // if ((num_goal_sets == 1) || (current_goal_set < num_goal_sets - 1)) {
        //     current_gains = gains1; // Use the first set of gains for all but the last goal
        // } 
        // else {
        //     current_gains = gains2; // Use the second set of gains for the last goal
        // }
        // if (err > 40) {
        //     current_gains = gains1;
        // }
        // else {
        //     current_gains = gains2; // Use the second set of gains for the last goal
        // }

        current_gains = gains1;

        // Convert gains to Eigen vector
        Eigen::VectorXf K(no_of_features);
        for(int i=0; i<no_of_features; i++){
            K[i] = current_gains[i];
        }

        std::cout << "Gain applied: " << K << std::endl;

        // Generate velocity
        // Convert qhat vector into matrix format
        Eigen::MatrixXf Qhat(no_of_features,no_of_actuators);

        int row_count = 0;
        int itr = 0;

        for (int i = 0; i < no_of_actuators; i++) {
                dr.push_back(0.0);
            }       

        while(row_count<no_of_features){
            for (int j = 0; j < no_of_actuators; j++){
                Qhat(row_count, j) = qhat[itr+j];
            }
            // Qhat.row(row_count) << qhat[itr], qhat[itr+1], qhat[itr+2]; //comment - possible change for no_of_actuators
            row_count = row_count + 1;
            itr = itr + no_of_actuators;  // comment - possible change for no_of_actuators
        }
        // std::cout<<"Created Jacobian: "<<Qhat<<std::endl;

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
        // std::cout<<"Published Jacobian Matrix: "<<Qhat_msg<<std::endl;
        
        // Publishing the Qhat matrix
        Qhat_pub.publish(Qhat_msg);

        Eigen::MatrixXf Qhat_T = Qhat.transpose();        

        Eigen::JacobiSVD<Eigen::MatrixXf> svd(Qhat, Eigen::ComputeFullU | Eigen::ComputeFullV);
        float tolerance = 1e-4; // Threshold for considering singular values as zero
        bool is_singular = svd.singularValues().minCoeff() < tolerance;

        // std::cout << "Min Coefficient: " << svd.singularValues().minCoeff() << std::endl;

        // Optional: Print the condition number for diagnostics
        float cond_number = svd.singularValues().maxCoeff() / svd.singularValues().minCoeff();
        // std::cout << "Condition number of Qhat: " << cond_number << std::endl;

        if (is_singular) {
            std::cerr << "Warning: Qhat is singular or close to singular!" << std::endl;
            status.data = -1;
            status_pub.publish(status);
            // Handle the singularity case, e.g., by applying regularization
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

        // std::cout<<"Jacobian: \n"<<Qhat<<std::endl;

        // Closed form solution for linearly independent columns
        // A_inv = (A.transpose()*A).inverse() * A.transpose()
        Eigen::MatrixXf Qhat_inv = (Qhat.transpose()*Qhat).inverse() * Qhat.transpose();
        // Eigen::MatrixXf Qhat_inv = (Qhat.transpose() * Qhat + reg_lambda * Eigen::MatrixXf::Identity(Qhat.cols(), Qhat.cols())).inverse() * Qhat.transpose();
        // std::cout << "Qhat_inv dimensions: " << Qhat_inv.rows() << "x" << Qhat_inv.cols() << std::endl;
        // std::cout<<"Inverted Jacobian: \n"<<Qhat_inv<<std::endl;
        // Saturating the error vector
        // for(int i=0; i<no_of_features; i++){
        //     if(abs(error_vec(i)) > saturation){
        //         error_vec(i) = (error_vec(i)/abs(error_vec(i)))*saturation;
        //     }
        // }
        // std::cout << "Error vector with no saturation: " << error_vec << std::endl;
        // IBVS control law (Velocity generator)
        //  With Berk 
        // P Control 
        // std::cout<<"Qhat_inv: "<<Qhat_inv<<std::endl;

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
        // std::cout<<"Published Jacobian Transpose Matrix: "<<Qhat_feat_msg<<std::endl;
        
        // Publishing the Qhat matrix
        Qhat_feat_pub.publish(Qhat_feat_msg);

        // std::cout<<"error_vec: "<<error_vec<<std::endl;
        // std::cout<<"gains: "<<lam<<std::endl;
        // joint_vel = lam*(Qhat_inv)*(error_vec);
        // std::cout<<"gains"<<Eigen::MatrixXf(K.asDiagonal())<<std::endl;
        // std::cout << "Diagonal gains matrix dimensions: " << Eigen::MatrixXf(K.asDiagonal()).rows() << "x" << Eigen::MatrixXf(K.asDiagonal()).cols() << std::endl;

        // Compute the error magnitude for adaptive gain
        float error_magnitude = unsaturated_error_vec.norm();

        // std::cout << "feature error norm: " << error_magnitude << std::endl;       
        
        // float adaptive_gain = p_lam * (1 + alpha_gains * error_magnitude); // Adaptive gain calculation

        float adaptive_gain = (1+(alpha_gains * error_magnitude));

        // std::cout << "Adaptive gain preprocess: " << adaptive_gain << std::endl;

        // if ((current_goal_set == num_goal_sets - 1) && (error_magnitude <= 40)){
        //     adaptive_gain = adaptive_gain * 2.5;
        // }
        // if (error_magnitude <= 40){
        //     adaptive_gain = (adaptive_gain + 1)/2.5;
        // }
        Eigen::VectorXf adaptive_gains(no_of_features);
        feature_errors.clear();
        for (int i = 0; i < no_of_features; i++) {
            float feature_error = abs(unsaturated_error_vec(i));       
            // std::cout << "individual_feature_error: " << feature_error << std::endl;     
            feature_errors.push_back(feature_error);
            adaptive_gains(i) = (alpha_gains * feature_error);
            // std::cout << "adaptive gain for feature: " << i << "::" << adaptive_gains(i) << std::endl;  
        }      


        std_msgs::Float32MultiArray feature_errors_msg;
        feature_errors_msg.data = feature_errors;

        // std::cout << "Diagonal adaptive gains: " << adaptive_gains << std::endl;

        Eigen::MatrixXf adaptive_gains_matrix = adaptive_gains.asDiagonal();

        // for (int i = 0; i < no_of_features; i++) {
        //     if (error_vec(i) > 0){
        //         K(i) = -K[i];
        //     }
        // }

        // std::cout << "gains after adjustment: " << K << std::endl;

        Eigen::MatrixXf K_diag = K.asDiagonal();

        // std::cout << "Inverse Jacobian: " << Qhat_inv << std::endl;

        // std::cout << "Adaptive gain postprocess: " << adaptive_gain << std::endl;

        // std::cout << "Applied gain: " << adaptive_gain * K_diag << std::endl;

        // std::cout << "Adaptive gains matrix: " << adaptive_gains_matrix << std::endl;

        joint_vel = (Qhat_inv)*(Eigen::MatrixXf(K.asDiagonal())*error_vec);
        // joint_vel = (Qhat_inv)*(adaptive_gains_matrix*K_diag)*(error_vec);
        // joint_vel = (Qhat_inv)*(adaptive_gains_matrix*error_vec);

        // std::cout << "Error and gain: " << (Eigen::MatrixXf(K.asDiagonal())*error_vec) << std::endl;


        // for (int i = 0; i < no_of_actuators; i++){
        //     if (joint_vel[i] >= 0.2){
        //         joint_vel[i] = (joint_vel[i]*0.5);
        //     }
        // }
        // float max_velocity = joint_vel.cwiseAbs().maxCoeff();
        // if (max_velocity > saturation) {
        //     joint_vel = joint_vel * (saturation / max_velocity);
        // }
        // std::cout << "Error_vec last element:"  <<  (error_vec(error_vec.size() - 1)) << std::endl; 
        // std::cout << "Error_vec second last element:"  <<  (error_vec(error_vec.size() - 2)) << std::endl; 
        // if ((error_vec(error_vec.size() - 1) < 0) && (error_vec(error_vec.size() - 2) >=0)){
        //     joint_vel[1] = (joint_vel[1]*2);
        //     joint_vel[2] = -joint_vel[2];
        // }

        // joint_vel = (Qhat_inv) * (adaptive_gains_matrix * K_diag) * (error_vec);

        // Implement adaptive gain adjustment without distorting the original control logic
        // for (int i = 0; i < joint_vel.size(); ++i) {
        //     if (abs(joint_vel[i]) < 0.001) { // Threshold to detect very slow movement
        //         joint_vel[i] *= 5.0; // Increase the velocity for the slower joint
        //     }
        // }

        // if (no_of_features == 8 || no_of_features == 10 || no_of_features == 12){
        //     joint_vel = (Qhat_inv)*(Eigen::MatrixXf(K.asDiagonal())*error_vec);
        //     // joint_vel = (Qhat_inv)*(adaptive_gain*K_diag)*(error_vec);
        //     // joint_vel = (Qhat_inv) * (adaptive_gains_matrix * K_diag) * (error_vec);
        // }        
        // else if (no_of_features==6){
        //     // joint_vel = lam*(Qhat_inv)*(error_vec);
        //     joint_vel = (Qhat_inv)*(Eigen::MatrixXf(K.asDiagonal())*error_vec);
        // } 
                
        // std::cout<<"joint_vel_1: "<<joint_vel[0]<<std::endl;
        // std::cout<<"joint_vel_2: "<<joint_vel[1]<<std::endl;
        // if (no_of_actuators==3){
        //     std::cout<<"joint_vel_3: "<<joint_vel[2]<<std::endl; 
        // }
        //uncomment - possible change for 3 joints
        // end of with Berk
        
        // std::cout<<"joint_vel: "<<joint_vel<<std::endl;
        // with Berk Sliding mode control
        // Eigen::VectorXf u_sliding_mode(no_of_features);
        // Eigen::VectorXf gain_sm_vec(no_of_features);
        // gain_sm_vec = gain_sm*(gain_sm_vec.setOnes(no_of_features));
        // // gain_sm_mat << 2*gain_sm, gain_sm, gain_sm, gain_sm;

        // for(int i=0; i<no_of_features; i++){
        //     u_sliding_mode[i] = gain_sm_vec[i]*sign(error_vec[i]);
        // }
        // joint_vel = Qhat_inv*u_sliding_mode;
        // end of with Berk Sliding mode control
/*      Working code for velocity scaling for results uptil 2-21-2022
        // Normalizing joint velocities
        float vel_sum = abs(joint_vel[0]) + abs(joint_vel[1]);
        // vel_sum is always +ve
        if(vel_sum > 0){
            joint_vel[0] = joint_vel[0]/(vel_sum);
            // std::cout<<"normalized joint1 vel:"<<joint_vel[0]<<"\n";
            joint_vel[0] = joint_vel[0]*(amplitude);
            // std::cout<<"capped normalized joint1 vel:"<<joint_vel[0]<<"\n";
            joint_vel[1] = joint_vel[1]/(vel_sum);
            // std::cout<<"normalized joint2 vel:"<<joint_vel[1]<<"\n";
            joint_vel[1] = joint_vel[1]*(amplitude);
            // std::cout<<"capped normalized joint2 vel:"<<joint_vel[1]<<"\n";
        }
End of working velocity scaling*/

        // Abhinav implementing a saturated P-controller 2-21-2022
        // This controller acts as SM controller for large errors
        // and converts to a P-controller closer to the ref
        // In the if blocks, first term determines the sign of the vel

/*        if(abs(joint_vel[0]) > amplitude){
            joint_vel[0] = (joint_vel[0]/abs(joint_vel[0])) * amplitude;
        }
        if(abs(joint_vel[1]) > amplitude){
            joint_vel[1] = (joint_vel[1]/abs(joint_vel[1])) * amplitude;
        }
*/
        // Publish velocity to robot

        // Check for NaNs in the computed velocities
        if (!joint_vel.allFinite()) {
            std::cerr << "Warning: NaN detected in joint velocities. Applying minimal velocities." << std::endl;
            // bool end_flag = true;
            // Apply a very small velocity to each joint as a fallback
            // joint_vel[0] = 0.001;
            // joint_vel[1] = 0.001;
            // joint_vel[2] = 0.0001;
            // No need to skip the iteration; we apply minimal velocities instead
        }

        j_vel.data.clear();
        j_vel.data.push_back(joint_vel[0]);
        j_vel.data.push_back(joint_vel[1]);
        if (no_of_actuators==3) {
            j_vel.data.push_back(joint_vel[2]); // Only add j3_vel if no_of_actuators is more than 2
        }
        
        // if ((error_magnitude <= 100) && (current_goal_set < (num_goal_sets - 1))){
        //     j_vel.data.clear();
        //     j_vel.data.push_back(joint_vel[0]);
        //     j_vel.data.push_back(joint_vel[1]);
        //     if (no_of_actuators==3) {
        //         j_vel.data.push_back(joint_vel[2]*0.1); // Only add j3_vel if no_of_actuators is more than 2
        //     }
        // }
        // // else if ((error_magnitude <= 70) && (current_goal_set == (num_goal_sets - 1))){
        // //     j_vel.data.clear();
        // //     j_vel.data.push_back(joint_vel[0]);
        // //     j_vel.data.push_back(joint_vel[1]);
        // //     if (no_of_actuators==3) {
        // //         j_vel.data.push_back(joint_vel[2]); // Only add j3_vel if no_of_actuators is more than 2
        // //     }
        // // }
        // else if ((error_magnitude <= 50) && (current_goal_set < (num_goal_sets - 1))){
        //     j_vel.data.clear();
        //     j_vel.data.push_back(joint_vel[0]);
        //     j_vel.data.push_back(joint_vel[1]);
        //         if (no_of_actuators==3) {
        //             j_vel.data.push_back(joint_vel[2]); // Only add j3_vel if no_of_actuators is more than 2
        //         }
        // }       
        // else if ((error_magnitude > 20) && (current_goal_set == (num_goal_sets - 1))){
        //     j_vel.data.clear();
        //     j_vel.data.push_back(joint_vel[0]);
        //     j_vel.data.push_back(joint_vel[1]);
        //     if (no_of_actuators==3) {
        //         j_vel.data.push_back(-joint_vel[2]); // Only add j3_vel if no_of_actuators is more than 2
        //     }
        // }
        // else if ((error_magnitude <= 20) && (current_goal_set == (num_goal_sets - 1))){
        //     j_vel.data.clear();
        //     j_vel.data.push_back(joint_vel[0]);
        //     j_vel.data.push_back(joint_vel[1]);
        //     if (no_of_actuators==3) {
        //         j_vel.data.push_back(joint_vel[2]); // Only add j3_vel if no_of_actuators is more than 2
        //     }
        // }
        // else {
        //     j_vel.data.clear();
        //     j_vel.data.push_back(joint_vel[0]);
        //     j_vel.data.push_back(joint_vel[1]);
        //     if (no_of_actuators==3) {
        //         j_vel.data.push_back(0); // Only add j3_vel if no_of_actuators is more than 2
        //     }
        // }
        
        // j_vel.data.clear();
        // j_vel.data.push_back(joint_vel[0]);
        // j_vel.data.push_back(joint_vel[1]);

        // // Add the third joint velocity based on the number of actuators and conditions
        // if (no_of_actuators == 3) {
        //     if (error_magnitude <= 100) {
        //         j_vel.data.push_back(joint_vel[2] * 0.1); // Scale down j3_vel if error_magnitude <= 100
        //     } else if ((error_magnitude <= 50) && (current_goal_set < num_goal_sets - 1)) {
        //         j_vel.data.push_back(joint_vel[2]); // Use full j3_vel if error_magnitude <= 50 and not the last goal set
        //     } else if ((error_magnitude <= 70) && (current_goal_set == num_goal_sets - 1)) {
        //         j_vel.data.push_back(joint_vel[2]); // Use full j3_vel if error_magnitude <= 70 and it's the last goal set
        //     } else {
        //         j_vel.data.push_back(0); // Set j3_vel to 0 for all other cases
        //     }
        // }

        // Check if we are in the goal switch pause period
        if (iterations_since_goal_change < zero_velocity_iterations) {
            j_vel.data.clear();
            for (int i = 0; i < no_of_actuators; ++i) {
                j_vel.data.push_back(0.0); // Insert zero velocity for each actuator
            }
            // Increment the counter
            iterations_since_goal_change++;
        } else {
            j_vel.data.clear();
            j_vel.data.push_back(joint_vel[0]);
            j_vel.data.push_back(joint_vel[1]);
            if (no_of_actuators == 3) {
                j_vel.data.push_back(joint_vel[2]); // Only add j3_vel if no_of_actuators is more than 2
            }
        }


        // std::cout<< "j1 vel: " << j_vel.data.at(0) << std::endl;
        // std::cout<< "j2 vel: " << j_vel.data.at(1) << std::endl;
        // std::cout<< "j3 vel: " << j_vel.data.at(2) << std::endl;

        std::cout<<"for error magnitude: " << error_magnitude <<" published joint_vel: "<< j_vel <<std::endl;

        j_pub.publish(j_vel);
        
        // Get current state of robot
        control_points.data.clear();
        // cp_client.call(cp_msg);
        kp_client.call(cp_msg);
        for(int i = 0; i<no_of_features; i++){
            // cur_features[i] = cp_msg.response.cp.data.at(i);
            cur_features[i] = cp_msg.response.kp.data.at(i);
            control_points.data.push_back(cur_features[i]);
        }

        // Compute change in state
        ds.clear();
        for(int i=0; i<no_of_features;i++){
            ds.push_back((cur_features[i]-old_features[i]));
        }
        
        // The += is not a bug, dr is set to 0 in the loop
        // Do not loose your mind every time you see this!
        dr[0] += joint_vel[0]*t;
        dr[1] += joint_vel[1]*t;
        if (no_of_actuators==3){
            dr[2] += joint_vel[2]*t; 
        }
        //comment/uncomment
    
        // Compute shape change magnitude
        float ds_accumulator = 0;
        for(int i = 0; i<no_of_features; i++){
            ds_accumulator += ds[i] * ds[i];
        }
        float ds_norm = sqrt(ds_accumulator);

        float err = sqrt(std::inner_product(error.begin(), error.end(), error.begin(), 0.0));
        // if (err < thresh) {
        //     ++current_goal_set; // Move to the next set of goal features
        //     // Publish the updated current goal set index
        //     current_goal_set_msg.data = current_goal_set;
        //     current_goal_set_pub.publish(current_goal_set_msg);
        //     if (current_goal_set < num_goal_sets) {
        //         old_features = cur_features;

        //         // Resetting change in joint angles and shape change vectors for the new goal
        //         std::fill(dSinitial.begin(), dSinitial.end(), 0);
        //         std::fill(dRinitial.begin(), dRinitial.end(), 0);
        //     }
        // }

        // Check if we should publish 0 velocities or calculated velocities
        // if (iterations_since_goal_change < zero_velocity_iterations) {
        //     // Publish 0 velocities
        //     j_vel.data.clear();
        //     for (int i = 0; i < no_of_actuators; ++i) {
        //         j_vel.data.push_back(0.0); // Insert zero velocity for each actuator
        //     }
        //     // Increment the counter
        //     iterations_since_goal_change++;
        // } else {
        //     // Publish calculated velocities
        //     j_vel.data.clear();
        //     j_vel.data.push_back(joint_vel[0]);
        //     j_vel.data.push_back(joint_vel[1]);
        //     j_vel.data.push_back(joint_vel[2]); // Adjust based on the number of actuators

        //     // Reset the counter if we just finished publishing zero velocities
        //     if (iterations_since_goal_change == zero_velocity_iterations) {
        //         iterations_since_goal_change = 0; // Reset counter
        //     }
        // }

        // j_pub.publish(j_vel);
        float current_thresh;
        // If the current goal is not the last goal the error thresh hold is 25 else it is 10 for now
        if (current_goal_set < num_goal_sets - 1) {
            current_thresh = thresh1; // Use thresh1 for all but the last goal
        } else {
            current_thresh = thresh2; // Use thresh2 for the last goal
        }
        std::cout<<"Current goal set "<<current_goal_set<< std::endl;
        std::cout<<"number of goal sets "<<num_goal_sets<< std::endl;
        std::cout<<"Current Threshold in use is "<<current_thresh<< std::endl;
        std::cout<<"Current Error is "<<err<<std::endl;

        // The following block is to change goals
        if (err < current_thresh) {  
            std::cout << "Goal " << current_goal_set << " reached. Moving to next goal." << std::endl;
            if (current_goal_set < num_goal_sets - 1) {
                ++current_goal_set; // Move to the next set of goal features
                // Publish the updated current goal set index                
                // print_fvector(goal_features);
                goal = goal_features[current_goal_set]; // Update the goal to the next set
                std::cout<<"new goal is"<<std::endl;
                print_fvector(goal);
                // Logging
                std::cout << "Switching to goal set " << current_goal_set << std::endl;

                // // Reset qhat to final_qhat_initial_estimation
                // qhat = final_qhat_initial_estimation; // Reset Jacobian to the initial estimation result
                // // it = 0; // Reset window iteration counter  
                // dSinitial.clear();
                // dRinitial.clear();          
                // // it = 0; // Reset window iteration counter
                // // Resetting change in joint angles and shape change vectors for the new goal
                // std::fill(dSinitial.begin(), dSinitial.end(), 0);
                // std::fill(dRinitial.begin(), dRinitial.end(), 0);
                std::cout << "Switched to goal set " << current_goal_set << std::endl; 

                // Calculate the error immediately after goal switch
                for(int i = 0; i < no_of_features; i++){
                    error[i] = cur_features[i] - goal[i];
                }

                Eigen::VectorXf error_vec(no_of_features);
                for(int i=0; i<no_of_features; i++){
                    error_vec(i) = error[i];
                }

                float err_mag = error_vec.norm();                

                std::cout << "error right after goal switch: " << err_mag << std::endl;

                // Resetting for every goal switch

                    // qhat = final_qhat_initial_estimation; // Resetting the Jacobian as the goal switched

                    // for(int i=0; i<no_of_features;i++){
                    //     dSinitial[i] = ds[i];
                    // }
                    // std::rotate(dSinitial.begin(), dSinitial.begin()+no_of_features, dSinitial.end());
                    // for(int i=0; i<no_of_actuators;i++){
                    //     dRinitial[i] = dr[i];
                    // }  
                    // std::rotate(dRinitial.begin(), dRinitial.begin()+no_of_actuators, dRinitial.end());
                    // dSmsg.data.clear();
                    // for(std::vector<float>::iterator itr = dSinitial.begin(); itr != dSinitial.end(); ++itr){
                    //     dSmsg.data.push_back(*itr);
                    // }
                    // dRmsg.data.clear();
                    // for(std::vector<float>::iterator itr = dRinitial.begin(); itr != dRinitial.end(); ++itr){
                    //     dRmsg.data.push_back(*itr);
                    // }
                    // // Push reset qhat to ROS Msg
                    // qhatmsg.data.clear();
                    // for(std::vector<float>::iterator itr = qhat.begin(); itr != qhat.end(); ++itr){
                    //     qhatmsg.data.push_back(*itr);
                    // }   
                    // // Print the contents of qhat
                    // std::cout << "The qhat inside goal switch: ";
                    // for (const auto& val : qhat) {
                    //     std::cout << val << " ";
                    // }
                    // std::cout << std::endl;
                    // // // Immediately call the service with the reset qhat
                    // msg.request.gamma_first_actuator = gamma1;
                    // msg.request.gamma_second_actuator = gamma2;
                    // msg.request.gamma_third_actuator = gamma3;
                    // msg.request.it = window-1;
                    // msg.request.dS = dSmsg;
                    // msg.request.dR = dRmsg;
                    // msg.request.qhat = qhatmsg;
                    // msg.request.feature_error_magnitude = error_magnitude;
                    // msg.request.feature_errors = feature_errors_msg;
                    // std::vector<float> qhatdot = msg.response.qhat_dot.data;
                    // for(int i = 0; i < qhat.size(); i++) {
                    //     qhat[i] = qhat[i] + qhatdot[i];
                    // }
                    // qhatmsg.data.clear();
                    // for(std::vector<float>::iterator itr = qhat.begin(); itr != qhat.end(); ++itr) {
                    //     qhatmsg.data.push_back(*itr);
                    // }    
                    // // Print the contents of qhat
                    // std::cout << "Updated qhat inside goal switch: ";
                    // for (const auto& val : qhat) {
                    //     std::cout << val << " ";
                    // }
                    // std::cout << std::endl;

                // Resetting only in case of large error after goal switch
                // if (err_mag > 250) {
                //     qhat = final_qhat_initial_estimation; // Resetting the Jacobian as the goal switched
                
                //     for(int i=0; i<no_of_features;i++){
                //         dSinitial[i] = ds[i];
                //     }
                //     std::rotate(dSinitial.begin(), dSinitial.begin()+no_of_features, dSinitial.end());
                //     for(int i=0; i<no_of_actuators;i++){
                //         dRinitial[i] = dr[i];
                //     }  
                //     std::rotate(dRinitial.begin(), dRinitial.begin()+no_of_actuators, dRinitial.end());

                //     dSmsg.data.clear();
                //     for(std::vector<float>::iterator itr = dSinitial.begin(); itr != dSinitial.end(); ++itr){
                //         dSmsg.data.push_back(*itr);
                //     }
                //     dRmsg.data.clear();
                //     for(std::vector<float>::iterator itr = dRinitial.begin(); itr != dRinitial.end(); ++itr){
                //         dRmsg.data.push_back(*itr);
                //     }

                //     // Push reset qhat to ROS Msg
                //     qhatmsg.data.clear();
                //     for(std::vector<float>::iterator itr = qhat.begin(); itr != qhat.end(); ++itr){
                //         qhatmsg.data.push_back(*itr);
                //     }   

                //     // Print the contents of qhat
                //     std::cout << "The qhat inside goal switch: ";
                //     for (const auto& val : qhat) {
                //         std::cout << val << " ";
                //     }
                //     std::cout << std::endl;

                //     // // Immediately call the service with the reset qhat
                //     msg.request.gamma_first_actuator = gamma1;
                //     msg.request.gamma_second_actuator = gamma2;
                //     msg.request.gamma_third_actuator = gamma3;
                //     msg.request.it = window-1;
                //     msg.request.dS = dSmsg;
                //     msg.request.dR = dRmsg;
                //     msg.request.qhat = qhatmsg;
                //     msg.request.feature_error_magnitude = error_magnitude;
                //     msg.request.feature_errors = feature_errors_msg;

                //     std::vector<float> qhatdot = msg.response.qhat_dot.data;

                //     for(int i = 0; i < qhat.size(); i++) {
                //         qhat[i] = qhat[i] + qhatdot[i];
                //     }

                //     qhatmsg.data.clear();
                //     for(std::vector<float>::iterator itr = qhat.begin(); itr != qhat.end(); ++itr) {
                //         qhatmsg.data.push_back(*itr);
                //     }    

                //     // Print the contents of qhat
                //     std::cout << "Updated qhat inside goal switch: ";
                //     for (const auto& val : qhat) {
                //         std::cout << val << " ";
                //     }
                //     std::cout << std::endl;

                //     }                

                // Pause for 2 seconds before continuing to the next goal
                // ros::Duration(2).sleep();
            } else {
                // All goals reached
                std::cout << "All goals reached" << std::endl;
                break;
            }
            // // // Publish the updated current goal set index
            // current_goal_set_msg.data = current_goal_set;
            // current_goal_set_pub.publish(current_goal_set_msg);
        }
        
        else{         
            // could change ds_norm to error_norm and stop updating near goal
        
            // Update sampling windows
            for(int i=0; i<no_of_features;i++){
                dSinitial[i] = ds[i];
            }
            std::rotate(dSinitial.begin(), dSinitial.begin()+no_of_features, dSinitial.end());
            for(int i=0; i<no_of_actuators;i++){
                dRinitial[i] = dr[i];
            }
            // dRinitial[0] = dr[0];
            // dRinitial[1] = dr[1];
            // dRinitial[2] = dr[2]; //
            std::rotate(dRinitial.begin(), dRinitial.begin()+no_of_actuators, dRinitial.end());
            
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

            // Print the contents of qhat
            std::cout << "The qhat outside goal switch: ";
            for (const auto& val : qhat) {
                std::cout << val << " ";
            }
            std::cout << std::endl;

            // populating request data
            // msg.request.gamma_general = gamma2;
            msg.request.gamma_first_actuator = gamma1;
            msg.request.gamma_second_actuator = gamma2;
            msg.request.gamma_third_actuator = gamma3;
            msg.request.it = window-1;
            msg.request.dS = dSmsg;
            msg.request.dR = dRmsg;
            msg.request.qhat = qhatmsg;
            msg.request.feature_error_magnitude = error_magnitude; 
            msg.request.feature_errors = feature_errors_msg; 
            // Call energy functional service
            energyClient.call(msg);
            // Populate service response
            std::vector<float> qhatdot = msg.response.qhat_dot.data;
            // Update Jacobian
            for(int i = 0; i<qhat.size(); i++){
                qhat[i] = qhat[i] + qhatdot[i]; // Updating each element of Jacobian
            }

            // Print the contents of qhat
            std::cout << "Updated qhat outside goal switch: ";
            for (const auto& val : qhat) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
            // Push updated Jacobian vector to ROS Msg
            qhatmsg.data.clear();
            for(std::vector<float>::iterator itr = qhat.begin(); itr != qhat.end(); ++itr){
                qhatmsg.data.push_back(*itr);
            }
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
        // for(int i=0; i < ds_msg.data.size(); i++) {
        //     std::cout << "ds_msg element " << i << " value: " << ds_msg.data[i] << std::endl;
        // }
        dr_msg.data.clear();
        for(int i=0; i<no_of_actuators; i++){
            dr_msg.data.push_back(dr[i]);
        }
        // dr_msg.data.clear();
        // dr_msg.data.push_back(dr[0]);
        // dr_msg.data.push_back(dr[1]);
        // dr_msg.data.push_back(dr[2]); //comment/uncomment on the basis of 
        dr.clear();
        ds_pub.publish(ds_msg);
        dr_pub.publish(dr_msg);        

        err_msg.data.clear();
        for(int i = 0; i<no_of_features;i++){
            err_msg.data.push_back(error[i]);
        }

        // publish
        std_msgs::Float32 J;
        J.data = msg.response.J;

        J_pub.publish(J);

        // Publish control points
        cp_pub.publish(control_points);

        // ROS_INFO("Publishing error message");
        // for(const auto& value : err_msg.data) {
        //     ROS_INFO("%f", value);
        // }
        
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

