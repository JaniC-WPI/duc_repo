import math
import numpy as np
from copy import deepcopy


class Kinematics():

    def __init__(self, DH_params_home, joint_var_types, mdh=False):
        """
        DH_params_home: DH parameters at home configuration
        joints_var_types: List of joint types from base to end-effector
            'theta': revolute joint
            'd': prismatic joint
            'alpha'
            'a'
        mdh: Modified DH or Standard DH
        """
        self.DH_params_home = np.array(deepcopy(DH_params_home))
        self.mdh = mdh  # Modified DH
        self.joint_var_types = joint_var_types
        self.joint_mat = []
        self.require_joint_padding = False  # Whether using intermediate frames
        # add joints' values to the DH table
        for j in joint_var_types:
            if j == 'theta':
                self.joint_mat.append([0, 0, 0, 1])
            elif j == 'd':
                self.joint_mat.append([0, 1, 0, 0])
            else:
                self.joint_mat.append([0, 0, 0, 0])
                self.require_joint_padding = True
        self.joint_mat = np.array(self.joint_mat)

    def forward(self, joint_vars):
        """
        Forward kinematics.
        """
        T_old = np.eye(4)
        # joint_vars_padded = []
        # if not self.require_joint_padding:
        #     joint_vars_padded = joint_vars
        # else:
        #     j = 0  # index of joint_vars
        #     for i in range(len(self.joint_var_types)):
        #         if self.joint_var_types[i] not in ['theta', 'd']:
        #             joint_vars_padded.append(0)
        #         else:
        #             joint_vars_padded.append(joint_vars[j])
        #             j += 1
        #
        DH_params = self.DH_params_home + \
            (np.diag(joint_vars) @ self.joint_mat)

        """
        T_cumulative
        ex: 5 th matrix in T_cumulative is transformation from link5 to link0

        T_individual
        ex: 5 th matrix in T_individual  is transformation from link5 to link 4
        """

        T_cumulative = []
        T_individual = []

        for joint_params in DH_params:
            T_new = self.t_matrix(joint_params)
            T_individual.append(T_new)
            T = T_old @ T_new
            T_cumulative.append(T)
            T_old = T

        return T_cumulative

    def t_matrix(self, joint_params):
        """
        Calculates transformation matrix from joint parameters.
        """
        a, d, alpha, theta = joint_params
        ct = math.cos(theta)
        st = math.sin(theta)
        ca = math.cos(alpha)
        sa = math.sin(alpha)

        if not self.mdh:
            """
            for classical DH parameters
            """
            T = np.array([[ct,   -st*ca,  st*sa,  a*ct],
                          [st,    ct*ca, -ct*sa,  a*st],
                          [0,       sa,     ca,     d],
                          [0,        0,      0,     1]])
        else:
            """
            For modified DH parameters
            to calculate transformation matrix from DH parameters
            """
            T = np.array([[ct,       -st,   0,      a],
                          [st*ca,  ct*ca, -sa,  -d*sa],
                          [st*sa,  ct*sa,  ca,   d*ca],
                          [0,          0,   0,      1]])

        return T


if __name__ == '__main__':
    DH_params_home = [
        [0,      0.333,             0,   0],
        [0,          0,  -(math.pi/2),   0],
        [0,      0.316,   (math.pi/2),   0],
        [0.0825,     0,   (math.pi/2),   0],
        [-0.0825, 0.384,  -(math.pi/2),  0],
        [0,          0,   (math.pi/2),   0],
        [0.088,      0,   (math.pi/2),   0],
        [0.1,      0.1070,   0,          0]]

    joint_types = ['theta', 'theta', 'theta', 'theta', 'theta',
                   'theta', 'theta', ' ']

    joint_angles = (0.040000191676486996, 0.015665356921891246, 0.004531416875103389, 0.20470613453454778, 0.0040978403291731524, -1.145515951463083, 0.0034248409741799435, -0.017448595150700186, -0.010164490335088594, 0)

    forward = Kinematics(DH_params_home, joint_types, mdh=True)
    ht = forward.forward(joint_angles[2:])

    print("len(ht)",len(ht))
    print(ht[-2])
