#!/usr/bin/env python3.8

import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray, Bool
import sys
# print(sys.path)

class UnscentedKalmanFilter(object):
    def __init__(self, dt, x_std_meas, y_std_meas, vx_std_meas, vy_std_meas):
        """
        :param dt: sampling time (time for 1 cycle)
        :param u_x: acceleration in x-direction
        :param u_y: acceleration in y-direction
        :param std_acc: process noise magnitude
        :param x_std_meas: standard deviation of the measurement in x-direction
        :param y_std_meas: standard deviation of the measurement in y-direction
        """
        # Counter to predict the first estimate
        self.count = 1
        # flag subscriber to stop correction        
        self.flag_sub = rospy.Subscriber("/feature/Flag", Bool, self.flag_cb, queue_size=1)
        self.do_correction = None        
        # Define sampling time
        self.dt = dt
        # Intial State
        self.X = np.matrix([[320], [240], [0.001], [0.001], [0.01], [0.01]])
        # declare states in prediction
        self.predicted_x = np.zeros(self.X.shape)
        # declare states after correction
        self.corrected_x = np.zeros(self.X.shape)
        # Define the State Transition Matrix A
        self.F = np.array([[1, 0, dt,  0, 0.5*dt**2, 0], 
              [0, 1,  0, dt,         0, 0.5*dt**2], 
              [0, 0,  1,  0,        dt,         0],
              [0, 0,  0,  1,         0,        dt],
              [0, 0,  0,  0,         1,         0],
              [0, 0,  0,  0,         0,         1]])  
        
        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0]])
        #Initial Process Noise Covariance
        self.Q = np.diag([50, 50, 50, 50, 3, 0.1])

        #Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas**2, 0, 0, 0],
                           [0, y_std_meas**2, 0, 0],
                           [0, 0, vx_std_meas**2, 0],
                           [0, 0, 0, vy_std_meas**2]])
        #Initial error Covariance Matrix
        self.P = np.diag([0.01, 0.01, 10, 10, 2, 2])

        # self.P = np.eye(6, 6)
        # self.Q = np.ones((6, 6))
        # self.R = np.ones((4, 4))
    
        # declare lists to average accelaration and velocity in prediction steps
        self.p_vel_list_x = []
        self.p_vel_list_y = []
        self.p_acc_list_x = []
        self.p_acc_list_y = []

        # declare lists to average accelaration and velocity in correction steps
        self.c_vel_list_x = []
        self.c_vel_list_y = []
        self.c_acc_list_x = []
        self.c_acc_list_y = []
        
        self.avg_length = 5
    
    def flag_cb(self, msg):
        self.do_correction = msg.data
        # print("skip correction?", self.do_correction)

    def predict(self):
    W = sigma_points(self.P, self.Q)
    X = WtoX(W, self.X)
    Y = XtoY(X, self.dt, np.array([self.X[4], self.X[5], 0]))
    self.X, self.rwvector = meanY(Y, self.X)
    self.P = calculatepk(W)
    return self.X

def update(self, z):
    Z = YtoZ(self.Y)
    zmean = np.mean(Z, axis=1)
    pzz = pzz(Z, zmean)
    pvv = pvv(pzz, self.R)
    pxz = pxz(W, Z, zmean)
    kk = kalman_gain(pxz, pvv)
    v = innovation(zmean, z)
    self.X = update_state(self.P, kk, v, self.X)
    self.P = update_cov(self.P, kk, pvv)
    return self.X

import numpy as np
import Quaternion


def sigma_points(P,Q):
    n = P.shape[0]
    s = np.linalg.cholesky(P+Q)
    s = s*np.sqrt(2*n)
    W = np.hstack((s,-s))
    return W

def WtoX(W,previous_state_x):
    qkminus1 = previous_state_x[:4]
    # W is 6*12
    qwvector = W
    #omegakminus1 = previous_state_x[4:]
    #omegaW = W[3:,:]
    X = np.zeros((7,4))
    for i in range(6):
        qw = Quaternion.vectoquat(qwvector[:,i])
        qkminus1qw = Quaternion.multiply_quaternion(qkminus1,qw)
        #omegakmius1plusomegaW = omegakminus1+omegaW[:,i]
        #X[:,i] = np.hstack((qkminus1qw, omegakmius1plusomegaW))
        X[i,:] = qkminus1qw
    X[6,:] = previous_state_x
    return X

def XtoY(X, deltat, omegak):
    Y = np.zeros((7,4))
    norm_omegak = np.linalg.norm(omegak)
    if norm_omegak == 0:
        qdelta = np.array([1,0,0,0])
    else:
        edelta = omegak*deltat
        qdelta = Quaternion.vectoquat(edelta)
    for i in range(7):
        Y[i,:] = Quaternion.multiply_quaternion(X[i,:], qdelta)
    return Y

def meanY(Y, previous_state_x):
    for j in range(100):
        evec = np.zeros((7,3))
        for i in range(7):
            prev_inv = Quaternion.inverse_quaternion(previous_state_x)
            ei = Quaternion.multiply_quaternion(Y[i,:], prev_inv)
            #print(ei)
            ei = Quaternion.normalize_quaternion(ei)
            eve = Quaternion.quattovec(ei)
            if np.linalg.norm(eve)==0:
                evec[i,:] = np.zeros((3,))
            else:
                evec[i,:] = (-np.pi + np.remainder(np.linalg.norm(eve)+np.pi,2*np.pi))/np.linalg.norm(eve)*eve
                #evec[i,:] = eve#/np.linalg.norm(eve)
        ei_avg = np.mean(evec, axis=0)

        previous_state_x = Quaternion.normalize_quaternion(Quaternion.multiply_quaternion(Quaternion.vectoquat(ei_avg), previous_state_x))
        if np.linalg.norm(ei_avg) < 0.01:
            # print(j)
            break

    return previous_state_x, evec

def calculatepk(W):
    pk = np.zeros((3,3))
    for i in range(7):
        pk = pk + W[i,:][:,None]*W[i,:][None,:]
    pk = pk/7
    return pk

def YtoZ(Y):
    Z = np.zeros((7,3))
    quatg = np.array([0,0,0,1])
    for i in range(7):
        qk = Y[i,:]
        qkinv = Quaternion.inverse_quaternion(qk)
        prod = Quaternion.multiply_quaternion(Quaternion.multiply_quaternion(qkinv,quatg),qk)
        Z[i,:] = Quaternion.quattovec(prod)
    #print(Z.T)
    return Z

def pzz(Z, zmean):
    pzz = np.zeros((3,3))
    Z_diff = Z - zmean
    for i in range(7):
        pzz = pzz + Z_diff[i,:][:,None]*Z_diff[i,:][None,:]
    pzz = pzz/7
    return pzz

def pvv(pzz, R):
    return pzz + R

def pxz(W, Z, zmean):
    pxz = np.zeros((3,3))
    Z_diff = Z-zmean
    for i in range(7):
        pxz = pxz + W[i,:][:,None]*Z_diff[i,:][None,:]
    pxz = pxz/7
    return pxz

def kalman_gain(pxz, pvv):
    return np.dot(pxz,np.linalg.inv(pvv))

def innovation(zmean, actual):
    return actual/np.linalg.norm(actual) - zmean

def update_state(pk,kk,v, ymean):
    return Quaternion.multiply_quaternion(Quaternion.vectoquat(np.dot(kk,v)),ymean)

def update_cov(pk, kk, pvv):
    return pk - np.dot(np.dot(kk,pvv),kk.T)


def run(self):
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        if self.do_correction is not None and self.do_correction:
            z = # get measurement from sensor
            self.update(z)
        else:
            self.predict()
        # publish the estimated state
        rate.sleep()

   
