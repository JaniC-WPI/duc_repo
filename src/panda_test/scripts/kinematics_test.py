#!/usr/bin/env python3

import math
import rospy
from sensor_msgs.msg import JointState
import tf
from scipy.spatial.transform import Rotation as R
from DH import Kinematics
from Robot import RobotTest, PandaTest, ScaraTest, UR5Test


class KinematicsTest():
    """
    Checks if kinematics is correct by comparing to tf.
    """

    def __init__(self, robot: RobotTest):
        self.robot = robot

        rospy.init_node('kinematics_test', anonymous=True)
        rospy.Subscriber(
            self.robot.joint_states_topic, JointState, self.calc_fk)
        self.tf_listener = tf.TransformListener()

    def calc_fk(self, joint_states: JointState):
        # calculates FK
        fk = self.robot.forward(joint_states.position)[-1]
        rot_quat = R.from_matrix(fk[:3, :3]).as_quat()

        # checks if calculated FK matches TF output
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                    self.robot.base_link, self.robot.target_link, rospy.Time())
            fk_tf = self.tf_listener.fromTranslationRotation(trans, rot)

            print()
            print('===============')
            print(fk)
            print('===============')
            print(fk_tf)
            print()
            print(rot_quat)
            print('===============')
            print(rot)
            print()
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException) as e:
            print(e)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    KinematicsTest(UR5Test()).run()
