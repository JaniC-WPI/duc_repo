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

bool computeEnergyFuncCallback(panda_test::energyFuncMsg::Request &req, panda_test::energyFuncMsg::Response &res){
    
    // std::cout <<"Inside Energy Callback"<<std::endl;
    // std::cout <<"window size: "<<window<<std::endl;
    // std::cout <<"no of features: "<<no_of_features<<std::endl;

    // Assign Request data
    // float gamma_general = req.gamma_general;
    float gamma_first_actuator = req.gamma_first_actuator; //Learning Rate
    float gamma_second_actuator = req.gamma_second_actuator;
    float gamma_third_actuator = req.gamma_third_actuator;
    float it = req.it; // iterator
    float feature_error_magnitude = req.feature_error_magnitude;


    std_msgs::Float32MultiArray dS =  req.dS;
    std_msgs::Float32MultiArray dR = req.dR;
    std_msgs::Float32MultiArray qhat = req.qhat;
    std_msgs::Float32MultiArray feature_errors_msg = req.feature_errors;    
    std::vector<float> feature_errors = feature_errors_msg.data;
    // std::cout << "assigned request data"<<std::endl;

    // Convert ROS MSG Arrays to Eigen Matrices
    
    //dS
    std::vector<float> dSdata = dS.data;
    // Declare dS matrix
    Eigen::MatrixXf dSmat(window,no_of_features);
    // Push data to dS matrix
    int row_count = 0;
    int itr = 0;
    while(row_count < window){
        for (int col = 0; col < no_of_features; col++) {
            dSmat(row_count, col) = dSdata[itr + col];
        }     
        // For 6 features
        // dSmat.row(row_count) << dSdata[itr], dSdata[itr+1], dSdata[itr+2], dSdata[itr+3], dSdata[itr+4], dSdata[itr+5];
        // For 8 features in 3d
        // dSmat.row(row_count) << dSdata[itr], dSdata[itr+1], dSdata[itr+2], dSdata[itr+3], dSdata[itr+4], dSdata[itr+5], dSdata[itr+6], dSdata[itr+7];
            // dSdata[itr+8], dSdata[itr+9];
        // std::cout<<"Pushing dS data to row:"<<row_count<<std::endl;
        itr = itr+no_of_features;
        row_count = row_count + 1;
    }
    // std::cout<<"Size of dSMat: "<<dSmat.rows()<<","<<dSmat.cols()<<std::endl;

    //dR
    std::vector<float> dRdata = dR.data;
    // Declare dR matrix
    Eigen::MatrixXf dRmat(window,no_of_actuators);
    // Push data to dR matrix
    row_count = 0;
    itr = 0;
    while(row_count < window){
        for (int j = 0; j < no_of_actuators; j++) {
            dRmat(row_count, j) = dRdata[itr + j];
        }
        // dRmat.row(row_count) << dRdata[itr], dRdata[itr+1], dRdata[itr+2];
        // std::cout<<"Pushing dR data to row:"<<row_count<<std::endl;
        itr = itr+no_of_actuators;
        row_count = row_count + 1;
    }
    // std::cout<<"Size of dRMat: "<<dRmat.rows()<<","<<dRmat.cols()<<std::endl;

    //qhat
    std::vector<float> qhatdata = qhat.data;
    // Declare qhat matrix
    Eigen::MatrixXf qhatMat(no_of_features,no_of_actuators); //comment - possible change for 3 joints
    // Push data to qhat matrix
    row_count = 0;
    itr = 0;
    while(row_count < no_of_features){
        for (int j = 0; j < no_of_actuators; j++) {
            qhatMat(row_count, j) = qhatdata[itr + j];
        }
        // qhatMat.row(row_count) << qhatdata[itr], qhatdata[itr+1], qhatdata[itr+2];
        // std::cout<<"Pushing qhat data to row:"<<row_count<<std::endl;
        itr = itr + no_of_actuators;
        row_count = row_count + 1;
    }
    // std::cout<<"Size of qhat: "<<qhatMat.rows()<<","<<qhatMat.cols()<<std::endl;
    // std::cout<<"Converted request data to ROS Msg"<<std::endl;

    // Compute Energy Functional
    Eigen::MatrixXf Ji = Eigen::MatrixXf::Zero(1,dSmat.cols());
    std_msgs::Float64MultiArray error_msg;
    error_msg.data.resize(dSmat.cols());
    // std::cout<<"Declared Ji"<<std::endl;
    
    for(int i=0; i<dSmat.cols();i++){
        // std::cout<<dSmat(it,i)<<std::endl;
        float cur_model_err = pow((dRmat.row(it)*qhatMat.row(i).transpose() - dSmat(it,i)),2);
        // std::cout<<"current model error:"<<cur_model_err<<std::endl;
        float old_err = pow((dRmat*qhatMat.row(i).transpose() - dSmat.col(i)).norm(),2);
        // std::cout<<"old err:"<<old_err<<std::endl;
        Ji(i) = (cur_model_err + old_err)/2;
        error_msg.data[i] = Ji(i);
        // std::cout<<"Ji:"<<Ji(i)<<std::endl;
    }
    model_error_pub.publish(error_msg);
    // std::cout<<"computed energy functional"<<std::endl;
    // Updated Jacobian Vectors
    for(int j = 0; j < no_of_actuators; j++){
    // Choose the appropriate learning rate based on the joint index
    // float current_gamma = (j < 2) ? gamma_general : gamma_third_actuator;
    float current_gamma = (j == 0) ? gamma_first_actuator : ((j == 1) ? gamma_second_actuator : gamma_third_actuator);

        for(int i=0; i<dSmat.cols(); i++){
            if(Ji(i) > eps){    // Update Jacobian if error greater than convergence threshold
                // float error_magnitude = Ji(i);
                // std::cout << "Latest Error: " << feature_error_magnitude << std::endl;

                // learning rate decreases on the basis of feature_error norm
                float adaptive_gamma = current_gamma / (1 + alpha_gamma * feature_error_magnitude);

                if (feature_error_magnitude > 100){
                    current_gamma = adaptive_gamma;
                }

                // std::cout << "Current Gamma" << current_gamma << std::endl;

                // learning rate on the basis of individual feature error
                float feature_error = feature_errors[i];
                // float adaptive_gamma = current_gamma / (1 + alpha_gamma * feature_error);

                // if (feature_error_magnitude <= 30){
                //     adaptive_gamma = 0.00001;
                // } 
                // float adaptive_gamma = current_gamma / (1 + alpha_gamma * error_magnitude); // Choose an appropriate alpha value
                // float adaptive_gamma = current_gamma / (error_magnitude); // Choose an appropriate alpha value
                // float adaptive_gamma = current_gamma / (feature_error_magnitude); // Choose an appropriate alpha value

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
                // Apply the selected gamma for this joint update
                qhatMat.row(i) = (-current_gamma*(H.transpose())*G).transpose();
                // qhatMat.row(i) = (-adaptive_gamma * (H.transpose()) * G).transpose();
                // if (feature_error_magnitude < 30){
                //     qhatMat.row(i) = (-0.00001 * (H.transpose()) * G).transpose();
                // }
                // else {
                //     qhatMat.row(i) = (-adaptive_gamma * (H.transpose()) * G).transpose();
                // }
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

        // Convert matrix to vector
        // for(int i = 0; i<qhatMat.rows(); i++){
        //     qhatMatVector.push_back(qhatMat(i,0));
        //     qhatMatVector.push_back(qhatMat(i,1));
        //     qhatMatVector.push_back(qhatMat(i,2)); //comment - possible change for no_of_actuators
        // }
        // std::cout<<"converted qhatmat to vector"<<std::endl;

        // Declare ROS Msg Array
        std_msgs::Float32MultiArray qhat_dotmsg;
        qhat_dotmsg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        qhat_dotmsg.layout.dim[0].label = "qhat_elements";
        qhat_dotmsg.layout.dim[0].size = qhatMatVector.size();
        qhat_dotmsg.layout.dim[0].stride = 1;
        qhat_dotmsg.data.clear();
        // std::cout<<"Declared ROS msg"<<std::endl;

        // Push data to ROS Msg Array
        for(std::vector<float>::iterator itr = qhatMatVector.begin(); itr != qhatMatVector.end(); ++itr){
            // std::cout <<*itr<<std::endl;
            qhat_dotmsg.data.push_back(*itr);
        }
        // std::cout<<"Qhat converted to ROS msg"<<std::endl;

    // Jacobian response msg
    res.qhat_dot = qhat_dotmsg;

    // std::cout<<"qhat_dot"<<qhat_dotmsg<<"\n";
    
    // Sum of Jis
    float J = 0.0;
    for(int i=0; i<dSmat.cols();i++){
        // std::cout<<"Individual Model Error?"<<Ji(i)<<std::endl;
        J = J + Ji(i);
    }
    // Assign Response data
    res.J = J;
    // std::cout<<"J Response "<<J<<"\n";
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
    
    // These are for adaptive VS
    // nh->getParam("vsbot/control/no_of_features", no_of_features);
    // nh->getParam("vsbot/control/no_of_actuators", no_of_actuators);

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

