#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Bool, Float64


class WorkspacePublisher:
    """
    Cover the workspace of a robot using velocity control.
    """

    def __init__(self, resolutions, joint_limits, v_max, timestep, sync=True):
        """
        resolutions: List of data intervals per joint (i.e. num_data_points-1)
        joint_limits: List of tuples representing lower and upper
            joint limits. E.g. [(min1, max1), (min2, max2), ...].
        v_max: Maximum velocity of each joint
        timestep: Default, will be used if not exceeding velocity limit.

        Joint order is taken into account for resolutions and joint_limits.
        The last joint is assumed to be the fastest.
        """
        self.resolutions = np.array(resolutions)
        self.joint_limits = joint_limits
        self.sync = sync

        self.num_joints = len(self.resolutions)

        # Full motion range of each joint
        self.motion_ranges = np.array([joint_limits[i][1] - joint_limits[i][0]
                                      for i in range(len(joint_limits))])

        # Determine velocities for joints 1..n
        #       v = q * data_rate / resolution
        self.velocities = np.minimum(
            self.motion_ranges / self.resolutions / timestep,
            v_max)
        self.time_steps = \
            self.motion_ranges / self.resolutions / self.velocities

        print(self.motion_ranges)
        print(self.time_steps)
        print(self.velocities)

        rospy.init_node('workspace_publisher', anonymous=True)

        self.triggered = False  # Whether the process is triggered
        rospy.Subscriber('/workspace_publisher/trigger', Bool,
                         self.trigger_publisher, queue_size=1)

        # Publishes when a movement is completed
        self.movement_done_pub = rospy.Publisher(
            '/workspace_publisher/movement_done', Bool, queue_size=1)

        # Publishes when the workspace is completed
        self.completed_pub = rospy.Publisher(
            '/workspace_publisher/completed', Bool, queue_size=1)

        self.pubs = [
            rospy.Publisher(
                f'/panda/joint{i*2}_velocity_controller/command',
                Float64, queue_size=1)
            for i in range(1, 4)
        ]

        self.counters = np.array([-1, -1, -1])
        self.counters_next = np.array([-1, -1, 0])

        rospy.sleep(2)

    def trigger_publisher(self, msg: Bool):
        if not self.triggered:
            self.triggered = msg.data

    def run(self):
        done = False
        rospy.loginfo('Workspace publisher running...')
        t = 1  # Iteration counter
        while not rospy.is_shutdown() and not done:
            rospy.loginfo(f'Iteration {t}')
            # Wait for trigger from kp_gen
            while self.sync and not self.triggered:
                pass
            self.triggered = False
            # Decide which joint to move this iteration
            # The first joint that has next counter different from counter
            # will move
            for i in range(len(self.counters)):
                if self.counters_next[i] != self.counters[i]:
                    break

            rospy.loginfo(f'\nMoving joint {i+1}')
            rospy.loginfo(
                f'Counters {self.counters}\tNext Counters {self.counters_next}')

            # Move the joint 1 step
            self.pubs[i].publish(Float64(self.velocities[i]))
            rospy.sleep(self.time_steps[i])
            self.pubs[i].publish(Float64(0))
            rospy.sleep(0.1)
            # Signal kp_gen node
            self.movement_done_pub.publish(Bool(True))

            # Increment counters
            # For resolutions R, the counter system is similar to 
            # base-R number system.
            self.counters[i] = self.counters_next[i]
            if i == self.num_joints - 1:  # When 
                self.counters_next[i] += 1
            for j in range(i, -1, -1):
                if self.counters_next[j] >= self.resolutions[j]:
                    self.counters_next[j] = 0
                    self.velocities[j] = -self.velocities[j]  # reverse the vel
                    if j > 0:
                        self.counters_next[j-1] += 1
                        if j != self.num_joints-1:
                            self.counters[j] = -1
                            self.counters_next[j] = -1
                    else:
                        done = True
                else:
                    break
            t += 1

        self.completed_pub.publish(Bool(True))
        rospy.loginfo('Workspace publisher completed...')
        rospy.spin()


if __name__ == '__main__':
    # Test
    resolutions = [15] * 3
    # joint_limits = [(-1.7628, 1.7628),
    #                 (-2.754, -0.075),
    #                 (-0.0175, 2.053)]
    joint_limits = [(-1.7628, 1.2),
                    (-2.754, -0.075),
                    (-0.0175, 1.7)]
    v_max = [2.1, 2.1, 2.58]
    timestep = 0.2
    WorkspacePublisher(resolutions, joint_limits, v_max, timestep, sync=False).run()
