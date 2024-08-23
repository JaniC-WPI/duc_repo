// This service computes updates for the estimated Jacobian by optimising an energy functional

#include "ros/ros.h"
#include <eigen3/Eigen/Dense>
#include <math.h>
#include <vector>

#include "panda_test/energyFuncMsg.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Float64MultiArray.h"
ros::NodeHandle* nh;

ros::Publisher model_error_pub;

int no_of_features; // ds column size
int window; // Estimation window size
float eps; // update threshold or convergence condition
int no_of_actuators; // qhat, dr column size
float alpha_gamma;
int num_goal_sets;
int debug_mode = 0;     

bool computeEnergyFuncCallback(panda_test::energyFuncMsg::Request &req, panda_test::energyFuncMsg::Response &res){
    // Assign Request data
    
    float gamma_first_actuator = req.gamma_first_actuator; //Learning Rate
    float gamma_second_actuator = req.gamma_second_actuator;
    float gamma_third_actuator = req.gamma_third_actuator;
    float it = req.it; // iterator
    float feature_error_magnitude = req.feature_error_magnitude;
    int data_size = req.data_size;

    std_msgs::Float32MultiArray dS =  req.dS;
    std_msgs::Float32MultiArray dR = req.dR;
    std_msgs::Float32MultiArray qhat = req.qhat;
    std_msgs::Float32MultiArray feature_errors_msg = req.feature_errors;    
    std::vector<float> feature_errors = feature_errors_msg.data;
    if(debug_mode == 1){
        std::cout << "assigned request data"<<std::endl;
    }

    if(debug_mode == 1){
        std::cout << "Current Window size in energy func: " << data_size << std::endl;
    }

    //dS
    std::vector<float> dSdata = dS.data;
    // Declare dS matrix
    Eigen::MatrixXf dSmat(data_size,no_of_features);
    // Push data to dS matrix
    int row_count = 0;
    int itr = 0;
    while(row_count < data_size){
        for (int col = 0; col < no_of_features; col++) {
            dSmat(row_count, col) = dSdata[itr + col];
        }     
        
        itr = itr+no_of_features;
        row_count = row_count + 1;
    }
    if(debug_mode == 1){
        std::cout<<"Size of dSMat: "<<dSmat.rows()<<","<<dSmat.cols()<<std::endl;
    }

    //dR
    std::vector<float> dRdata = dR.data;
    // Declare dR matrix
    Eigen::MatrixXf dRmat(data_size,no_of_actuators);
    // Push data to dR matrix
    row_count = 0;
    itr = 0;
    while(row_count < data_size){
        for (int j = 0; j < no_of_actuators; j++) {
            dRmat(row_count, j) = dRdata[itr + j];
        }

        itr = itr+no_of_actuators;
        row_count = row_count + 1;
    }
    if(debug_mode == 1){
        std::cout<<"Size of dRMat: "<<dRmat.rows()<<","<<dRmat.cols()<<std::endl;
    }

    //qhat
    std::vector<float> qhatdata = qhat.data;
    // Declare qhat matrix
    Eigen::MatrixXf qhatMat(no_of_features,no_of_actuators); 
    // Push data to qhat matrix
    row_count = 0;
    itr = 0;
    while(row_count < no_of_features){
        for (int j = 0; j < no_of_actuators; j++) {
            qhatMat(row_count, j) = qhatdata[itr + j];
        }
        
        itr = itr + no_of_actuators;
        row_count = row_count + 1;
    }
    if(debug_mode == 1){
        std::cout<<"Size of qhat: "<<qhatMat.rows()<<","<<qhatMat.cols()<<std::endl;
    }
    // Compute Energy Functional
    Eigen::MatrixXf Ji = Eigen::MatrixXf::Zero(1,dSmat.cols());
    std_msgs::Float64MultiArray error_msg;
    error_msg.data.resize(dSmat.cols());
    
    for(int i=0; i<dSmat.cols();i++){
        float cur_model_err = pow((dRmat.row(it) * qhatMat.row(i).transpose() - dSmat((it), i)), 2);
        float old_err = pow((dRmat*qhatMat.row(i).transpose() - dSmat.col(i)).norm(),2);
        Ji(i) = (cur_model_err + old_err)/2;
        error_msg.data[i] = Ji(i);
    }
    model_error_pub.publish(error_msg);
    if(debug_mode == 1){
        std::cout<<"computed energy functional"<<std::endl;
    }

    // Updated Jacobian Vectors
    for(int j = 0; j < no_of_actuators; j++){
    // Choose the appropriate learning rate based on the joint index
    float current_gamma = (j == 0) ? gamma_first_actuator : ((j == 1) ? gamma_second_actuator : gamma_third_actuator);

        for(int i=0; i<dSmat.cols(); i++){
            if(Ji(i) > eps){    // Update Jacobian if error greater than convergence threshold
                // learning rate decreases on the basis of feature_error norm
                float adaptive_gamma = current_gamma / (1 + alpha_gamma * feature_error_magnitude);

                // learning rate on the basis of individual feature error
                float feature_error = feature_errors[i];
                 // Choose an appropriate alpha value
                // std::cout << "fixed learning rate: " << current_gamma << std::endl;
                // std::cout << "latest adaptive learning rate: " << adaptive_gamma << " for feature " << feature_error_magnitude << std::endl;
                Eigen::MatrixXf G1 = dRmat*(qhatMat.row(i).transpose()) - dSmat.col(i);
                float G2 = dRmat.row(it)*(qhatMat.row(i).transpose()) - dSmat(it,i);
                Eigen::MatrixXf G ((G1.rows()+1),1);
                G << G1,
                     G2;

                Eigen::MatrixXf H1 (no_of_actuators,(window+1));
                H1 << dRmat.transpose(), dRmat.row(it).transpose();

                Eigen::MatrixXf H = H1.transpose(); 
                // Apply the selected gamma for this joint update - I am keeping the next commenetd line as I want to test between fixed and adaptive lr
                // qhatMat.row(i) = (-current_gamma*(H.transpose())*G).transpose();
                qhatMat.row(i) = (-adaptive_gamma * (H.transpose()) * G).transpose();                
            }
        }        
    }

        // Convert Eigen::Matrix to ROS MSG Array
        // Declare vector to store qhatMat elements
        std::vector<float> qhatMatVector;

        for (int i = 0; i < qhatMat.rows(); i++) {
            for (int j = 0; j < no_of_actuators; j++) {
                qhatMatVector.push_back(qhatMat(i,j));
             }
        }  

        // Declare ROS Msg Array
        std_msgs::Float32MultiArray qhat_dotmsg;
        qhat_dotmsg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        qhat_dotmsg.layout.dim[0].label = "qhat_elements";
        qhat_dotmsg.layout.dim[0].size = qhatMatVector.size();
        qhat_dotmsg.layout.dim[0].stride = 1;
        qhat_dotmsg.data.clear();

        // Push data to ROS Msg Array
        for(std::vector<float>::iterator itr = qhatMatVector.begin(); itr != qhatMatVector.end(); ++itr){
            qhat_dotmsg.data.push_back(*itr);
        }

    // Jacobian response msg
    res.qhat_dot = qhat_dotmsg;
    
    // Sum of Jis
    float J = 0.0;
    for(int i=0; i<dSmat.cols();i++){
        J = J + Ji(i);
    }
    // Assign Response data
    res.J = J;
    return true;
}

int main(int argc, char **argv){
    // ROS Initialization
    ros::init(argc, argv, "energy_functional");
    nh = new ros::NodeHandle;
    std::cout <<"Starting Energy Functional Service" <<std::endl;
    
    nh->getParam("vsbot/estimation/window", window); //size of estimation window
    nh->getParam("vsbot/estimation/epsilon", eps);
    nh->getParam("vsbot/control/alpha_gamma", alpha_gamma);
    nh->getParam("vsbot/control/debug_mode", debug_mode);

    // These are for shape based VS
    nh->getParam("vsbot/shape_control/no_of_features", no_of_features);
    nh->getParam("vsbot/shape_control/no_of_actuators", no_of_actuators);
    nh->getParam("dl_controller/num_goal_sets", num_goal_sets);

    model_error_pub = nh->advertise<std_msgs::Float64MultiArray>("individual_model_error", 1);

    // Declare Service Server
    ros::ServiceServer compute_energy_func = nh->advertiseService("computeEnergyFunc", computeEnergyFuncCallback);

    ros::spin();
    return 0;
}

