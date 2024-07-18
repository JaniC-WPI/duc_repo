#include "ros/ros.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>  // For std::inner_product

#include "panda_test/energyFuncMsg.h"
#include "panda_test/dl_img.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"
#include "std_msgs/Int32.h"
#include "std_msgs/Bool.h"

// Fuzzy logic control variables
float membershipFunction(float x, float center, float width) {
    return exp(-pow((x - center) / width, 2));
}

float fuzzyRule(float e, float de) {
    // Define membership functions for error and change in error
    float neg_large = membershipFunction(e, -1.0, 0.5) * membershipFunction(de, -1.0, 0.5);
    float neg_small = membershipFunction(e, -0.5, 0.5) * membershipFunction(de, -0.5, 0.5);
    float zero = membershipFunction(e, 0.0, 0.5) * membershipFunction(de, 0.0, 0.5);
    float pos_small = membershipFunction(e, 0.5, 0.5) * membershipFunction(de, 0.5, 0.5);
    float pos_large = membershipFunction(e, 1.0, 0.5) * membershipFunction(de, 1.0, 0.5);

    return (-2.0 * neg_large + -1.0 * neg_small + 0.0 * zero + 1.0 * pos_small + 2.0 * pos_large) /
           (neg_large + neg_small + zero + pos_small + pos_large);
}

void fuzzyControlLoop(
    const Eigen::VectorXf& error_vec,
    const Eigen::VectorXf& prev_error_vec,
    Eigen::MatrixXf& Qhat_inv,
    Eigen::VectorXf& joint_vel,
    const Eigen::VectorXf& gains,
    float lam)
{
    // Compute change in error (delta error)
    Eigen::VectorXf delta_error_vec = error_vec - prev_error_vec;

    // Compute fuzzy gain adjustment factor for each feature
    Eigen::VectorXf gain_adjustment_factors(error_vec.size());
    for (int i = 0; i < error_vec.size(); ++i) {
        float e = error_vec[i];
        float de = delta_error_vec[i];
        gain_adjustment_factors[i] = fuzzyRule(e, de);
    }

    // Compute the adaptive gain based on the fuzzy logic output
    Eigen::VectorXf adaptive_gains = lam * (Eigen::VectorXf::Ones(error_vec.size()) + gain_adjustment_factors.cwiseProduct(gains));

    // Define negative gains for K
    Eigen::VectorXf K = -gains;

    // Compute the joint velocities using the adaptive gain and the saturated error vector
    joint_vel = adaptive_gains.asDiagonal() * (Qhat_inv * (K.asDiagonal() * error_vec));
}

int main(int argc, char **argv) {
    // ROS initialization
    ros::init(argc, argv, "shape_servo_control_node");
    ros::NodeHandle n;

    // Initialize ROS publishers
    ros::Publisher j_pub = n.advertise<std_msgs::Float64MultiArray>("joint_vel", 1);
    ros::Publisher ds_pub = n.advertise<std_msgs::Float64MultiArray>("ds_record", 1);
    ros::Publisher dr_pub = n.advertise<std_msgs::Float64MultiArray>("dr_record", 1);
    ros::Publisher J_pub = n.advertise<std_msgs::Float32>("J_modelerror", 1);
    ros::Publisher err_pub = n.advertise<std_msgs::Float64MultiArray>("servoing_error", 1);
    ros::Publisher status_pub = n.advertise<std_msgs::Int32>("vsbot/status", 1);
    ros::Publisher cp_pub = n.advertise<std_msgs::Float64MultiArray>("vsbot/control_points", 1);
    ros::Publisher current_goal_set_pub = n.advertise<std_msgs::Int32>("current_goal_set_topic", 1);

    // Initializing ROS subscribers
    bool end_flag = false; // true when servoing is completed. Triggered by user
    bool start_flag = false; // true when camera stream is ready
    auto start_flag_callback = [&](const std_msgs::Bool &msg) { start_flag = msg.data; };
    auto end_flag_callback = [&](const std_msgs::Bool &msg) { end_flag = msg.data; };
    ros::Subscriber end_flag_sub = n.subscribe("vsbot/end_flag", 1, end_flag_callback);
    ros::Subscriber start_flag_sub = n.subscribe("franka/control_flag", 1, start_flag_callback);

    while (!start_flag) {
        ros::Duration(10).sleep();
        ros::spinOnce();
    }

    ros::Duration(10).sleep();

    // Initializing service clients
    ros::service::waitForService("computeEnergyFunc", 1000);
    ros::ServiceClient energyClient = n.serviceClient<panda_test::energyFuncMsg>("computeEnergyFunc");
    ros::service::waitForService("franka_kp_dl_service", 1000);
    ros::ServiceClient kp_client = n.serviceClient<panda_test::dl_img>("franka_kp_dl_service");

    // Load parameters
    int window, no_of_features, no_of_actuators;
    float lam, alpha, regularization_lambda, saturation;
    std::vector<float> gains;
    n.getParam("vsbot/estimation/window", window);
    n.getParam("vsbot/control/no_of_features", no_of_features);
    n.getParam("vsbot/control/no_of_actuators", no_of_actuators);
    n.getParam("vsbot/control/lam", lam);
    n.getParam("vsbot/control/alpha", alpha);
    n.getParam("vsbot/control/regularization_lambda", regularization_lambda);
    n.getParam("vsbot/control/saturation", saturation);
    n.getParam("vsbot/control/gains", gains);

    Eigen::VectorXf gains_vec = Eigen::VectorXf::Map(gains.data(), gains.size());

    // Initialize variables for error and control loop
    Eigen::VectorXf error_vec(no_of_features);
    Eigen::VectorXf prev_error_vec = Eigen::VectorXf::Zero(no_of_features);
    Eigen::MatrixXf Qhat_inv(no_of_features, no_of_actuators); // Example dimensions
    Eigen::VectorXf joint_vel(no_of_actuators);
    std::vector<float> current_features(no_of_features, 0);
    std::vector<float> old_features(no_of_features, 0);

    // Obtain initial robot state and goal features
    std::vector<std::vector<float>> goal_features;
    int num_goal_sets;
    n.getParam("dl_controller/num_goal_sets", num_goal_sets);
    goal_features.resize(num_goal_sets);
    for (int i = 0; i < num_goal_sets; ++i) {
        std::string param_name = "dl_controller/goal_features" + std::to_string(i + 1);
        n.getParam(param_name, goal_features[i]);
    }

    std::vector<float> goal = goal_features[0]; // Start with the first goal set
    int current_goal_set = 0; // Index of the current goal feature set

    // Initial estimation period
    status.data = 1;
    status_pub.publish(status);

    ros::Rate r(10); // Example rate
    int it = 0;
    while (it < window) {
        // Simulate small displacements and record dS and dR
        // For simplicity, assume some small displacement mechanism here

        // Obtain current robot state
        panda_test::dl_img cp_msg;
        kp_client.call(cp_msg);
        for (int i = 0; i < no_of_features; i++) {
            current_features[i] = cp_msg.response.kp.data[i];
        }

        // Compute dS and dR
        // For simplicity, assume some mechanism to compute dS and dR

        // Store current state as old state for next iteration
        old_features = current_features;

        // Increase iterator
        it++;
        ros::spinOnce();
        r.sleep();
    }

    // Start servoing period
    status.data = 2;
    status_pub.publish(status);
    ros::Rate control_r(10); // Example control rate

    while (ros::ok() && !end_flag) {
        // Obtain current robot state
        panda_test::dl_img cp_msg;
        kp_client.call(cp_msg);
        for (int i = 0; i < no_of_features; i++) {
            current_features[i] = cp_msg.response.kp.data[i];
        }

        // Compute error vector
        for (int i = 0; i < no_of_features; ++i) {
            error_vec(i) = current_features[i] - goal[i];
        }

        // Regularized Jacobian inverse computation
        Qhat_inv = (Qhat.transpose() * Qhat + regularization_lambda * Eigen::MatrixXf::Identity(Qhat.cols(), Qhat.cols())).inverse() * Qhat.transpose();

        // Fuzzy control logic
        fuzzyControlLoop(error_vec, prev_error_vec, Qhat_inv, joint_vel, gains_vec, lam);

        // Apply joint velocities to the robot
        std_msgs::Float64MultiArray joint_vel_msg;
        joint_vel_msg.data.clear();
        for (int i = 0; i < joint_vel.size(); ++i) {
            joint_vel_msg.data.push_back(joint_vel[i]);
        }
        j_pub.publish(joint_vel_msg);

        // Store current error as previous error for next iteration
        prev_error_vec = error_vec;

        // Update goal set if necessary
        float error_norm = error_vec.norm();
        float current_thresh = (current_goal_set < num_goal_sets - 1) ? 0.1 : 0.05; // Example thresholds
        if (error_norm < current_thresh) {
            if (current_goal_set < num_goal_sets - 1) {
                current_goal_set++;
                goal = goal_features[current_goal_set];
            } else {
                end_flag = true;
            }
        }

        // Publish control points and status
        std_msgs::Float64MultiArray control_points_msg;
        control_points_msg.data.clear();
        for (const auto &feature : current_features) {
            control_points_msg.data.push_back(feature);
        }
        cp_pub.publish(control_points_msg);

        status_pub.publish(status);
        ros::spinOnce();
        control_r.sleep();
    }

    // Commanding 0 velocity to robot
    std_msgs::Float64MultiArray stop_vel_msg;
    stop_vel_msg.data.clear();
    for (int i = 0; i < no_of_actuators; ++i) {
        stop_vel_msg.data.push_back(0.0);
    }
    j_pub.publish(stop_vel_msg);

    // Final status update
    status.data = -1;
    status_pub.publish(status);

    ros::spin();
    return 0;
}