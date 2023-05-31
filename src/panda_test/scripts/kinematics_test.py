#!/usr/bin/env python3

import math
import rospy
from sensor_msgs.msg import JointState
import tf
from scipy.spatial.transform import Rotation as R
from DH import Kinematics


# Panda
# DH_params_home = [
#     [0,      0.333,             0,   0],
#     [0,          0,  -(math.pi/2),   0],
#     [0,      0.316,   (math.pi/2),   0],
#     [0.0825,     0,   (math.pi/2),   0],
#     [-0.0825, 0.384,  -(math.pi/2),  0],
#     [0,          0,   (math.pi/2),   0],
#     [0.088,      0,   (math.pi/2),   0]]
#     # [0.1,      0.1070,   0,          0]]
# joint_types = ['t', 't', 't', 't', 't', 't', 't']
# mdh = True
# joint_states_topic = '/panda/joint_states'
# base_link = '/panda_link0'
# target_link = '/panda_link7'

# SCARA
DH_params_home = [
    [0,       0.05, 0,        0],
    [0.425,   0.45, 0,        0],
    [0.345,  -0.06, 3.14,  0]]
joint_types = ['t', 't', 'd']
mdh = False
joint_states_topic = '/scara/joint_states'
base_link = '/base_link'
target_link = '/link3'

# ABB IRB2400
# DH_params_home = [
#     [0, 0, 0, 0],
#     [0.097, 0.44, -math.pi/2, 0],
#     [0.705, 0, 0, 0],
#     [0.135, 0.264, -math.pi/2, 0],
#     [0, 0.497, math.pi/2, 0],
#     [0, 0.065, -math.pi/2, 0]]
# joint_types = ['t', 't', 't', 't', 't', 't']
# mdh = True
# joint_states_topic = '/joint_states'
# base_link = '/base_link'
# target_link = '/link_6'

# Fanuc LR Mate 200iC
# DH_params_home = [
#     [0.075, 0.33, -math.pi/2, 0],
#     [0.3, 0, math.pi, -math.pi/2],
#     [-0.075, 0, math.pi/2, math.pi],
#     [0, -0.32, -math.pi/2, 0],
#     [0, 0, math.pi/2, 0],
#     [0, -0.08, math.pi, math.pi]]
# joint_types = ['t', 't', 't', 't', 't', 't']
# mdh = False
# joint_states_topic = '/joint_states'
# base_link = '/base_link'
# target_link = '/link_6'


class KinematicsTest():

    def __init__(self):
        rospy.init_node('kinematics_test', anonymous=True)
        rospy.Subscriber(joint_states_topic, JointState, self.calc_fk)
        self.tf_listener = tf.TransformListener()
        self.kinematics = Kinematics(DH_params_home, joint_types, mdh=mdh)

    def calc_fk(self, joint_states: JointState):
        # calculates FK

        # Panda
        # fk = self.kinematics.forward(joint_states.position[2:])[-1]

        # All other robots
        fk = self.kinematics.forward(joint_states.position)[-1]
        rot_quat = R.from_matrix(fk[:3, :3]).as_quat()

        # checks if calculated FK matches TF output
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                    base_link, target_link, rospy.Time())
            fk_tf = self.tf_listener.fromTranslationRotation(trans, rot)

            print()
            # print(fk)
            print(rot_quat)
            print('===============')
            # print(fk_tf)
            print(rot)
            print()
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException) as e:
            print(e)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    KinematicsTest().run()
