#!/usr/bin/env python3.8

import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray, Bool

class UnscentedKalmanFilter(object):
    def __init__(self, dt, x_std_meas, y_std_meas, vx_std_meas, vy_std_meas):
        """
        :param dt: sampling time (time for 1 cycle)
        :param x_std_meas: standard deviation of the measurement in x-direction
        :param y_std_meas: standard deviation of the measurement in y-direction
        :param vx_std_meas: standard deviation of the velocity measurement in x-direction
        :param vy_std_meas: standard deviation of the velocity measurement in y-direction
        """
        # ROS setup
        self.flag_sub = rospy.Subscriber("/feature/Flag", Bool, self.flag_cb, queue_size=1)
        self.do_correction = None
        
        # Time step
        self.dt = dt
        
        # Dimensions
        self.n_x = 6  # State dimension
        self.n_z = 4  # Measurement dimension
        
        # UKF parameters
        self.alpha = 0.001  # Small positive value (1e-4 to 1)
        self.ki = 0.0      # kappa, usually 0 or 3-n
        self.beta = 2.0    # For Gaussian distributions
        
        # Calculate lambda and scaling parameter
        self.lambda_ = self.alpha * self.alpha * (self.n_x + self.ki) - self.n_x
        self.n_sigma = 2 * self.n_x + 1
        self.gamma = np.sqrt(self.n_x + self.lambda_)
        
        # Initialize weights
        self._init_weights()
        
        # Initialize state vector [x, y, vx, vy, ax, ay]
        self.x = np.array([[320.0], 
                          [240.0], 
                          [0.001], 
                          [0.001], 
                          [0.01], 
                          [0.01]])
        
        # Initialize covariance matrices
        self.P = np.diag([0.01, 0.01, 10.0, 10.0, 2.0, 2.0])
        self.Q = np.diag([50.0, 50.0, 50.0, 50.0, 3.0, 0.1])
        self.R = np.diag([x_std_meas**2, y_std_meas**2, 
                         vx_std_meas**2, vy_std_meas**2])
        
        # Initialize sigma points matrix
        self.sigma_points = np.zeros((self.n_x, self.n_sigma))

    def _init_weights(self):
        """Initialize UKF weights for mean and covariance"""
        self.weights_m = np.zeros(self.n_sigma)
        self.weights_c = np.zeros(self.n_sigma)
        
        # Weight for mean and covariance of central sigma point
        self.weights_m[0] = self.lambda_ / (self.n_x + self.lambda_)
        self.weights_c[0] = self.weights_m[0] + (1 - self.alpha**2 + self.beta)
        
        # Weights for remaining sigma points
        for i in range(1, self.n_sigma):
            self.weights_m[i] = 1.0 / (2 * (self.n_x + self.lambda_))
            self.weights_c[i] = self.weights_m[i]

    def _generate_sigma_points(self):
        """Generate sigma points using current state and covariance"""
        # Calculate square root of P using Cholesky decomposition
        L = np.linalg.cholesky((self.n_x + self.lambda_) * self.P)
        
        # Set central sigma point
        self.sigma_points[:, 0] = self.x.flatten()
        
        # Set remaining sigma points
        for i in range(self.n_x):
            self.sigma_points[:, i+1] = self.x.flatten() + L[i]
            self.sigma_points[:, i+1+self.n_x] = self.x.flatten() - L[i]
            
        return self.sigma_points

    def _state_transition(self, sigma_point):
        """
        Predict sigma points using the nonlinear state transition function
        State vector: [x, y, vx, vy, ax, ay]
        """
        x, y, vx, vy, ax, ay = sigma_point
        
        # Nonlinear state equations
        x_pred = x + vx * self.dt + 0.5 * ax * self.dt**2
        y_pred = y + vy * self.dt + 0.5 * ay * self.dt**2
        vx_pred = vx + ax * self.dt
        vy_pred = vy + ay * self.dt
        ax_pred = ax  # Assuming constant acceleration
        ay_pred = ay
        
        return np.array([x_pred, y_pred, vx_pred, vy_pred, ax_pred, ay_pred])

    def _measurement_function(self, sigma_point):
        """
        Convert predicted sigma points to measurement space
        Measurement vector: [x, y, vx, vy]
        """
        return np.array([sigma_point[0],   # x position
                        sigma_point[1],    # y position
                        sigma_point[2],    # x velocity
                        sigma_point[3]])   # y velocity

    def flag_cb(self, msg):
        """ROS callback for correction flag"""
        self.do_correction = msg.data

    def predict(self):
        """UKF prediction step"""
        # Generate sigma points
        self._generate_sigma_points()
        
        # Predict sigma points
        chi_pred = np.zeros_like(self.sigma_points)
        for i in range(self.n_sigma):
            chi_pred[:, i] = self._state_transition(self.sigma_points[:, i])
        
        # Predict mean
        self.x = np.zeros((self.n_x, 1))
        for i in range(self.n_sigma):
            self.x += self.weights_m[i] * chi_pred[:, i].reshape(-1, 1)
        
        # Predict covariance
        self.P = np.zeros((self.n_x, self.n_x))
        for i in range(self.n_sigma):
            diff = (chi_pred[:, i].reshape(-1, 1) - self.x)
            self.P += self.weights_c[i] * np.dot(diff, diff.T)
        
        # Add process noise
        self.P += self.Q
        
        # Update sigma points for next step
        self.sigma_points = chi_pred
        
        return self.x

    def update(self, z):
        """UKF update step"""
        if not self.do_correction:
            return self.x
            
        # Transform sigma points to measurement space
        z_sigma = np.zeros((self.n_z, self.n_sigma))
        for i in range(self.n_sigma):
            z_sigma[:, i] = self._measurement_function(self.sigma_points[:, i])
        
        # Predict measurement
        z_pred = np.zeros((self.n_z, 1))
        for i in range(self.n_sigma):
            z_pred += self.weights_m[i] * z_sigma[:, i].reshape(-1, 1)
        
        # Calculate innovation covariance
        S = np.zeros((self.n_z, self.n_z))
        Pxz = np.zeros((self.n_x, self.n_z))
        
        for i in range(self.n_sigma):
            z_diff = z_sigma[:, i].reshape(-1, 1) - z_pred
            x_diff = self.sigma_points[:, i].reshape(-1, 1) - self.x
            
            S += self.weights_c[i] * np.dot(z_diff, z_diff.T)
            Pxz += self.weights_c[i] * np.dot(x_diff, z_diff.T)
        
        # Add measurement noise
        S += self.R
        
        # Kalman gain
        K = np.dot(Pxz, np.linalg.inv(S))
        
        # Update state and covariance
        innovation = z - z_pred
        self.x = self.x + np.dot(K, innovation)
        self.P = self.P - np.dot(K, np.dot(S, K.T))
        
        return self.x


## USAGE:

# # Initialize
# ukf = UnscentedKalmanFilter(
#     dt=0.1,
#     x_std_meas=0.1,
#     y_std_meas=0.1,
#     vx_std_meas=0.1,
#     vy_std_meas=0.1
# )

# # Use in loop
# predicted_state = ukf.predict()
# measurement = np.array([[x], [y], [vx], [vy]])  # Your measurements
# corrected_state = ukf.update(measurement)
