#include "ros/ros.h"
#include <eigen3/Eigen/Dense>
#include <math.h>
#include <iostream>
#include <vector>

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
    std::cout<<"Compute Energy Func is waited"<<std::endl;
    // ros::service::waitForService("franka_control_service", 1000);       
                                // this service generates control points
    ros::service::waitForService("franka_kp_dl_service", 1000);  
    std::cout<<"franka_kp_service is waited"<<std::endl;
                                // this service genarates key_points for dream

    // ros::service::waitForService("binary_image_output", 1000);

    ros::ServiceClient energyClient = n.serviceClient<panda_test::energyFuncMsg>("computeEnergyFunc");
    std::cout<<"Compute Energy func is getting called ?"<<std::endl;
    // ros::ServiceClient cp_client = n.serviceClient<encoderless_vs::franka_control_points>("franka_control_service");

    // Added client for dream kp generation
    ros::ServiceClient kp_client = n.serviceClient<panda_test::dl_img>("franka_kp_dl_service");
    std::cout<<"franka kp service is getting called ?"<<std::endl;

    // Initializing status msg
    std_msgs::Int32 status;
    status.data = 0;
    status_pub.publish(status);

    // Servoing variables
    int window; // Estimation window size
    n.getParam("vsbot/estimation/window", window);

    float rate; // control & estimation loop rate
    n.getParam("vsbot/estimation/rate", rate);

    // float control_rate;
    // n.getParam("vsbot/control/rate", control_rate);

    float thresh;
    n.getParam("vsbot/control/thresh",thresh);

    float lam;
    n.getParam("vsbot/control/lam",lam);

    std::vector<float> gains;
    n.getParam("vsbot/control/gains",gains);

    float gain_sm;
    n.getParam("vsbot/control/gain_sm", gain_sm);

    n.getParam("vsbot/shape_control/no_of_features", no_of_features);
    n.getParam("vsbot/shape_control/no_of_actuators", no_of_actuators);

    std::vector<float> goal (no_of_features,0);
    n.getParam("dl_controller/goal_features", goal);
    // print_fvector(goal);

    int it = 0;                                     // iterator
    std::vector<float> error (no_of_features,0);    //error vector
    float err = 0.0;                                // error norm
    std_msgs::Float64MultiArray err_msg;            // feature error
    // std::cout << "Initialized Servoing Variables" << std::endl;

    // Estimation variables
    float gamma; // learning rate
    n.getParam("vsbot/estimation/gamma", gamma);
    
    float gamma2; // learning rate during control loop
    n.getParam("vsbot/estimation/gamma2", gamma2);

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

    // std::vector<float> qhat {7,13,-21,17,19,-23,11,-29}; // ,4.7,-13,27,31}; // initial Jacobian matrix as vector
    // std::vector<float> qhat {7,13,-21,17,19,-23,11,-29,31,11,3,17};
                        //  12 elements in qhat(6x2) for 6 elements in features and 2 joints
    // std::vector<float> qhat {7,13,-21,17,19,-23,11,-29,31,11,3,17,9,12,3,5,-14,20};
                        // 18 Elements in qhat(6X3) for 6 elements in feature vector and 3 joints or actuators
    // std::vector<float> qhat {7,13,-21,17,19,-23,11,-29,31,11,3,17,9,12,3,5,-14,20,9,20,-34,21,-6,18};
                        // 24 Elements in qhat(8X3) for 8 elements in feature vector and 3 joints or actuators
    // std::vector<float> qhat {7,13,-21,17,19,-23,11,-29,31,11,3,17,9,12,3,5,-14,20,9,20,-34,21,-6,18,11,-9,34,43,25,16};
                        // 30 Elements in qhat(10X3) for 10 elements in feature vector and 3 joints or actuators
    // std::vector<float> qhat {7,13,-21,17,19,-23,11,-29,31,11,3,17,9,12,3,5,-14,20,9,20,-34,21,-6,18,11,-9,34,43,25,16,9,65,43,-21,-8,-10};
    // std::vector<float> qhat {7,13,-21,17,19,-23,11,-29,31,11,3,17,14,5,8,1};
                        //  16 elements in qhat(8x2) for 6 elements in features
    // std::vector<float> qhat {7,13,-21,17,19,-23,11,-29,31,11,3,17,14,5,8,1,19, 71};
                        //  18 elements in qhat(8x2) for 4 points that is 8 kp elements and depth feature as the 9th element in features
    
    // std::vector<float> qhat {7,13,-21,17,19,-23,11,-29,31,11,3,17,1.3,0.9};
    // std::vector<float> qhat {0.07,0.13,-0.21,0.17,0.19,-0.23,0.11,-0.29};
    // std::vector<float> qhat {.7,1.3,-.21,.17,-0.01,-0.5,-0.2,-0.3};
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
    // Convert gains to Eigen vector
    // Eigen::VectorXf K(no_of_features);
    // for(int i=0; i<no_of_features; i++){
    //     K[i] = gains[i];
    // }


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
    
    // Collecting data for estimation window
    while (it < window){

        // Publish sin vel to both joints
        float j1_vel = amplitude*sin(param);
        float j2_vel = amplitude*cos(param); 
        float j3_vel = amplitude*(cos(param)+sin(param));  // comment out when 3rd joint not in use
        // float j1_vel = amplitude*sin(param);
        // float j2_vel = amplitude*cos(param); 
        
        param = param + 0.1;
        
        // Adding noise to sinusoidal velocity
        // j1_vel.data += (2*((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - 0.5));
        // j2_vel.data += (2*((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) - 0.5));
        
        j_vel.data.clear();
        j_vel.data.push_back(j1_vel);
        j_vel.data.push_back(j2_vel);
        j_vel.data.push_back(j3_vel); // comment out when 3rd joint not in use

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
        dr.push_back((j3_vel*t)); // comment out when 3rd joint not in use

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
    j_vel.data.push_back(0.0); //comment/uncomment depending on number of joints

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

    // Compute Jacobian
    it = 0;
    panda_test::energyFuncMsg msg;
    while(it < window){
        // Service request data
        msg.request.gamma = gamma;
        msg.request.it = it;
        msg.request.dS = dSmsg;
        msg.request.dR = dRmsg;
        msg.request.qhat = qhatmsg;

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
    
    // Switching to control loop rate
    t = 1/rate;
    ros::Rate control_r{rate};

    // Publish status
    status.data = 2;
    status_pub.publish(status);

    // Refresh subscribers
    ros::spinOnce();


    // ----------------------- Control Loop for Servoing -----------------------------
    while(!end_flag){    // convergence condition
        // error norm "err" is always positive
        
        // compute current error & norm
        // print_fvector(cur_features);
        // print_fvector(goal);
        for(int i=0; i<no_of_features;i++){
            error[i] = cur_features[i] - goal[i];
        }
    
        float err_acc = 0; // accumulator vairable for computing error norm
        for(int i=0; i<no_of_features; i++){
            err_acc += error[i]*error[i];
        }
        err = sqrt(err_acc);
        err_acc = 0; // Reset error accumulator
        // std::cout<<" norm:"<<err<<std::endl;

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
        
        // Convert error std::vector to Eigen::vector
        // for matrix computations
        Eigen::VectorXf error_vec(no_of_features);
        for(int i=0; i<no_of_features; i++){
            error_vec(i) = error[i];
        }

        // joint velocity Eigen::vector
        // Eigen::Vector2f joint_vel; // uncomment for 2 joints

        Eigen::Vector3f joint_vel;  // uncomment for 3 joints

        // std::cout<<"Jacobian: \n"<<Qhat<<std::endl;

        // Closed form solution for linearly independent columns
        // A_inv = (A.transpose()*A).inverse() * A.transpose()
        Eigen::MatrixXf Qhat_inv = (Qhat.transpose()*Qhat).inverse() * Qhat.transpose();
        // std::cout<<"Inverted Jacobian: \n"<<Qhat_inv<<std::endl;
        // Saturating the error vector
        for(int i=0; i<no_of_features; i++){
            if(abs(error_vec(i)) > saturation){
                error_vec(i) = (error_vec(i)/abs(error_vec(i)))*saturation;
            }
        }
        // IBVS control law (Velocity generator)
        //  With Berk 
        // P Control 
        std::cout<<"Qhat_inv: "<<Qhat_inv<<std::endl;
        std::cout<<"error_vec: "<<error_vec<<std::endl;
        std::cout<<"gains: "<<lam<<std::endl;
        joint_vel = lam*(Qhat_inv)*(error_vec);
        // joint_vel = (Qhat_inv)*(Eigen::MatrixXf(K.asDiagonal())*error_vec);
        std::cout<<"joint_vel_1: "<<joint_vel[0]<<std::endl;
        std::cout<<"joint_vel_2: "<<joint_vel[1]<<std::endl;
        std::cout<<"joint_vel_3: "<<joint_vel[2]<<std::endl; //uncomment - possible change for 3 joints
        // end of with Berk
        
        
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
        j_vel.data.clear();
        j_vel.data.push_back(joint_vel[0]);
        j_vel.data.push_back(joint_vel[1]);
        j_vel.data.push_back(joint_vel[2]); //comment/uncomment depending on # of joints

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
        dr[2] += joint_vel[2]*t; //comment/uncomment
    
        // Compute shape change magnitude
        float ds_accumulator = 0;
        for(int i = 0; i<no_of_features; i++){
            ds_accumulator += ds[i] * ds[i];
        }
        float ds_norm = sqrt(ds_accumulator);

        if(err > thresh){         
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
            // populating request data
            msg.request.gamma = gamma2;
            msg.request.it = window-1;
            msg.request.dS = dSmsg;
            msg.request.dR = dRmsg;
            msg.request.qhat = qhatmsg;
            // Call energy functional service
            energyClient.call(msg);
            // Populate service response
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
                ds_msg.data.push_back(ds[i]);
            }

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
        }

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
        
        err_pub.publish(err_msg);
        
        
        // Publish status msg
        status_pub.publish(status);

        // Refresh subscriber callbacks
        ros::spinOnce();
        control_r.sleep();
    }

    // Commanding 0 velocity to robot 
    j_vel.data.clear();
    j_vel.data.push_back(0.0);
    j_vel.data.push_back(0.0);
    j_vel.data.push_back(0.0); // comment/ uncomment on the basis of joints

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

