import rospy
import numpy as np
import tf
import random
import math
from std_msgs.msg import Float64, Bool, Float64MultiArray
from DH import Kinematics
from sensor_msgs.msg import JointState


class RobotTest:
    """
    Collection of robot-specific parameters and test functions.
    """

    # TODO: Add robot name/class name
    def __init__(self, DH_params_home, joint_types, mdh,
                 joint_states_topic, base_link, target_link,
                 kine_init=True,
                 frames=None,
                 duplicated_joints=None,

                 fk_use_tf=False,
                 tf_listener=None,
                 # Workspace covering-related
                 initial_vel_sign=None,

                 # Kp Gen Test-related
                 func_init=None, func_pre=None,
                 func_post=None, func_end=None):
        """
        DH_params_home: 2D list representing Home DH parameters
        joint_types: List of joint names from base to end-effector
            'theta': revolute joint
            'd': prismatic joint
            'alpha'
            'a'
        joint_states_topic: Name of the joint_states topic
        base_link: Base link name
        target_link: Target link name

        kine_init: Bool : If False, do not initialize a Kinematics object

        frames: String : List of all frames of the robot to be included in FK

        duplicated_joints: List of joint indices (0..n-1) that duplicates (
            at the same position) with other joint(s)

        func_init: Runs once before all tests
        func_pre: Runs before every test
        func_post: Runs after every test
        func_end: Runs once after all tests
        """
        # TODO: Make func_* abstract methods
        self.kine_init = kine_init
        if kine_init:
            self.kinematics = Kinematics(DH_params_home, joint_types, mdh=mdh)
        else:
            self.kinematics = None
        self.num_joints = len(DH_params_home)

        self.frames = frames
        self.duplicated_joints = duplicated_joints

        self.initial_vel_sign = initial_vel_sign

        self.fk_use_tf = fk_use_tf
        self.tf_listener = None

        self.func_init = func_init  # Runs once before tests
        self.func_pre = func_pre  # Runs before every test
        self.func_post = func_post  # Runs after every test
        self.func_end = func_end  # Runs once after all tests

        # Workspace

    def process_joint_pos(self, joint_pos):
        """
        Preprocess joint positions if needed.
        """
        return joint_pos

    def forward(self, joint_vars):
        """
        Calculates forward kinematics (4x4 TF matrix). It uses either
        DH::forward or tf package.
        """
        if self.kine_init:
            processed_joint_vars = self.process_joint_pos(joint_vars)
            if self.fk_use_tf and self.frames is not None:
                ret = []
                for f in self.frames:
                    (trans, rot) = self.tf_listener.lookupTransform(
                            self.base_link, f, rospy.Time())
                    ret.append(
                        self.tf_listener.fromTranslationRotation(trans, rot))
                return ret
            return self.kinematics.forward(processed_joint_vars)
        return None

    def ws_publish_joint(self, i, val):
        """
        ***For Workspace plotting***
        Publishes [val] to joint [i]th.
        """
        pass

    def ws_get_num_joints(self):
        """
        ***For Workspace plotting***
        Returns number of joints to plot workspace.
        """
        pass


class PandaTest2D(RobotTest):

    DH_params_home = [
        [0,      0.333,             0,   0],
        [0,          0,  -(math.pi/2),   0],
        [0,      0.316,   (math.pi/2),   0],
        [0.0825,     0,   (math.pi/2),   0],
        [-0.0825, 0.384,  -(math.pi/2),  0],
        [0,          0,   (math.pi/2),   0],
        [0.088,      0,   (math.pi/2),   0]]
        # [0.1,      0.1070,   0,          0]]
    joint_types = ['theta'] * 7
    mdh = True
    joint_states_topic = '/panda/joint_states'
    base_link = '/panda_link0'
    target_link = '/panda_link7'
    duplicated_joints = [1, 6]
    joint_limits = [(-2.8973, 2.8973),
                    (-1.7628, 1.7628),
                    (-2.8973, 2.8973),
                    (-3.0718, -0.0698),
                    (-2.8973, 2.8973),
                    (-0.0175, 3.7525),
                    (-2.8973, 2.8973)]

    def __init__(self):
        super().__init__(
            self.DH_params_home,
            self.joint_types,
            self.mdh,
            self.joint_states_topic,
            self.base_link,
            self.target_link,

            duplicated_joints=self.duplicated_joints,

            func_init=self.func_init_wsp,
            # func_pre=self.func_pre_panda_vel,
            func_post=self.func_post_wsp,
            # func_end=self.func_pre_panda_vel
        )

        self.ws_joints = [2, 4, 6]
        # Joint publishers
        self.pubs = [
            rospy.Publisher(
                f'/panda/joint{j}_velocity_controller/command',
                Float64, queue_size=1)
            for j in self.ws_joints
        ]

        # Communication with workspace_publisher
        self.wsp_trigger_pub = rospy.Publisher(
            '/workspace_publisher/trigger', Bool, queue_size=1)

    def process_joint_pos(self, joint_pos):
        return joint_pos[2:]

    def ws_publish_joint(self, i, val):
        """
        ***For Workspace plotting***
        Publishes [val] to joint [i]th. (0-based)
        """
        self.pubs[i].publish(Float64(val))

    def ws_get_num_joints(self):
        """
        ***For Workspace plotting***
        Returns number of joints to plot workspace.
        """
        return len(self.ws_joints)

    def func_post_panda_pos(self):
        # Post-test function (runs after every test)
        # moves the robot to a random configuration with joints 2 and 4
        pub2 = rospy.Publisher(
                "/panda/joint2_position_controller/command", Float64, queue_size=1)
        pub4 = rospy.Publisher(
                "/panda/joint4_position_controller/command", Float64, queue_size=1)
        joint2 = random.uniform(-1, 1)
        joint4 = random.uniform(-1, 1)
        pub2.publish(Float64(joint2))
        pub4.publish(Float64(joint4))
        # rospy.sleep(1)  # wait for the joints to reach targets

    def func_post_panda_vel(self):
        # Post-test function (runs after every test)
        # moves the robot to a random configuration with joints 2, 4, 6

        for i in range(3):
            # New velocity has opposite sign with previous value to prevent
            # reaching joint limits
            self.joints[i] = \
                math.copysign(random.uniform(-1, 1),
                              -self.joints[i])
            self.pubs[i].publish(
                Float64(self.joints[i]))

    def func_post_panda_vel_2(self):
        # Post-test function (runs after every test)
        # Covers the whole workspace
        pass

    def joint_vel_pub(self, joint_vel):
        """
        Publishes list of velocity to corresponding selected joints.
        """
        for i in range(len(joint_vel)):
            self.pubs[i].publish(Float64(joint_vel[i]))

    def stop_joints(self):
        # stops the joints
        self.joint_vel_pub([0] * self.ws_get_num_joints())

    def func_init_wsp(self):
        # TODO: fix this!!!
        # Move selected joints to lower limit positions
        self.joint_vel_pub([-1] * self.ws_get_num_joints())
        rospy.sleep(2)
        self.stop_joints()

    def func_post_wsp(self):
        self.wsp_trigger_pub.publish(Bool(True))


class PandaReal2D(RobotTest):

    DH_params_home = [
        [0,      0.333,             0,   0],
        [0,          0,  -(math.pi/2),   0],
        [0,      0.316,   (math.pi/2),   0],
        [0.0825,     0,   (math.pi/2),   0],
        [-0.0825, 0.384,  -(math.pi/2),  0],
        [0,          0,   (math.pi/2),   0],
        [0.088,      0,   (math.pi/2),   0]]
        # [0.1,      0.1070,   0,          0]]
    joint_types = ['theta'] * 7
    mdh = True
    joint_states_topic = '/joint_states'
    base_link = '/panda_link0'
    target_link = '/panda_link7'
    duplicated_joints = [1, 6]
    joint_limits = [(-0, 0),
                    # (-1.7628, 1.2),
                    (-1.2628, 1.2628),
                    (-0, 0),
                    (-2.4708, -1.0708),
                    # (-1.5481, -0.775),
                    (-0, 0),
                    # (-0.0175, 1.75),
                    (1.1951, 2.9021),
                    (-0, 0)]

    def __init__(self):
        super().__init__(
            self.DH_params_home,
            self.joint_types,
            self.mdh,
            self.joint_states_topic,
            self.base_link,
            self.target_link,

            duplicated_joints=self.duplicated_joints,
            initial_vel_sign = np.array([1, -1, -1]),

            kine_init=True,

            func_init=self.func_init_wsp,
            # func_pre=self.func_pre_panda_vel,
            func_post=self.func_post_wsp,
            # func_end=self.func_pre_panda_vel
        )

        self.ws_joints = [2, 4, 6]
        # Joint publishers
        self.pub = rospy.Publisher(
            '/joint_group_velocity_controller/command',
            Float64MultiArray, queue_size=1)

        # Communication with workspace_publisher
        self.wsp_trigger_pub = rospy.Publisher(
            '/workspace_publisher/trigger', Bool, queue_size=1)

    def process_joint_pos(self, joint_pos):
        return joint_pos[:-2]

    def ws_publish_joint(self, i, val):
        """
        ***For Workspace plotting***
        Publishes [val] to joint [i]th. (0-based)
        """
        vel = [0] * 7
        vel[self.ws_joints[i]-1] = val
        self.joint_vel_pub(vel)

    def ws_get_num_joints(self):
        """
        ***For Workspace plotting***
        Returns number of joints to plot workspace.
        """
        return len(self.ws_joints)

    def joint_vel_pub(self, joint_vel):
        msg = Float64MultiArray()
        msg.data = joint_vel
        self.pub.publish(msg)

    def stop_joints(self):
        # stops the joints
        self.joint_vel_pub([0] * 7)

    def func_init_wsp(self):
        pass

    def func_post_wsp(self):
        self.wsp_trigger_pub.publish(Bool(True))


class PandaTest3D(RobotTest):

    DH_params_home = [
        [0,      0.333,             0,   0],
        [0,          0,  -(math.pi/2),   0],
        [0,      0.316,   (math.pi/2),   0],
        [0.0825,     0,   (math.pi/2),   0],
        [-0.0825, 0.384,  -(math.pi/2),  0],
        [0,          0,   (math.pi/2),   0],
        [0.088,      0,   (math.pi/2),   0]]
        # [0.1,      0.1070,   0,          0]]
    joint_types = ['theta'] * 7
    mdh = True
    joint_states_topic = '/panda/joint_states'
    base_link = '/panda_link0'
    target_link = '/panda_link7'
    duplicated_joints = [1, 6]
    joint_limits = [(-2.8973, 2.8973),
                    (-1.7628, 1.7628),
                    (-2.8973, 2.8973),
                    (-3.0718, -0.0698),
                    (-2.8973, 2.8973),
                    (-0.0175, 3.7525),
                    (-2.8973, 2.8973)]

    def __init__(self):
        super().__init__(
            self.DH_params_home,
            self.joint_types,
            self.mdh,
            self.joint_states_topic,
            self.base_link,
            self.target_link,

            duplicated_joints=self.duplicated_joints,

            func_init=self.func_init_wsp,
            # func_pre=self.func_pre_panda_vel,
            func_post=self.func_post_wsp,
            # func_end=self.func_pre_panda_vel
        )

        # self.ws_joints = [2, 4, 6]
        self.ws_joints = [1, 2, 3, 4, 5, 6]
        # Joint publishers
        self.pubs = [
            rospy.Publisher(
                f'/panda/joint{j}_velocity_controller/command',
                Float64, queue_size=1)
            for j in self.ws_joints
        ]

        # Communication with workspace_publisher
        self.wsp_trigger_pub = rospy.Publisher(
            '/workspace_publisher/trigger', Bool, queue_size=1)

    def process_joint_pos(self, joint_pos):
        return joint_pos[2:]

    def ws_publish_joint(self, i, val):
        """
        ***For Workspace plotting***
        Publishes [val] to joint [i]th. (0-based)
        """
        self.pubs[i].publish(Float64(val))

    def ws_get_num_joints(self):
        """
        ***For Workspace plotting***
        Returns number of joints to plot workspace.
        """
        return len(self.ws_joints)

    def func_post_panda_pos(self):
        # Post-test function (runs after every test)
        # moves the robot to a random configuration with joints 2 and 4
        pub2 = rospy.Publisher(
                "/panda/joint2_position_controller/command", Float64, queue_size=1)
        pub4 = rospy.Publisher(
                "/panda/joint4_position_controller/command", Float64, queue_size=1)
        joint2 = random.uniform(-1, 1)
        joint4 = random.uniform(-1, 1)
        pub2.publish(Float64(joint2))
        pub4.publish(Float64(joint4))
        # rospy.sleep(1)  # wait for the joints to reach targets

    def func_post_panda_vel(self):
        # Post-test function (runs after every test)
        # moves the robot to a random configuration with joints 2, 4, 6

        for i in range(3):
            # New velocity has opposite sign with previous value to prevent
            # reaching joint limits
            self.joints[i] = \
                math.copysign(random.uniform(-1, 1),
                              -self.joints[i])
            self.pubs[i].publish(
                Float64(self.joints[i]))

    def func_post_panda_vel_2(self):
        # Post-test function (runs after every test)
        # Covers the whole workspace
        pass

    def joint_vel_pub(self, joint_vel):
        """
        Publishes list of velocity to corresponding selected joints.
        """
        for i in range(len(joint_vel)):
            self.pubs[i].publish(Float64(joint_vel[i]))

    def stop_joints(self):
        # stops the joints
        self.joint_vel_pub([0] * self.ws_get_num_joints())

    def func_init_wsp(self):
        # TODO: fix this!!!
        # Move selected joints to lower limit positions
        self.joint_vel_pub([-1] * self.ws_get_num_joints())
        rospy.sleep(2)
        self.stop_joints()

    def func_post_wsp(self):
        self.wsp_trigger_pub.publish(Bool(True))


class ScaraTest(RobotTest):

    DH_params_home = [
        [0,       0.05, 0,        0],
        [0.425,   0.45, 0,        0],
        [0.345,  -0.06, 3.14,  0]]
    joint_types = ['theta', 'theta', 'd']
    mdh = False
    joint_states_topic = '/scara/joint_states'
    base_link = '/base_link'
    target_link = '/link3'

    def __init__(self):
        super().__init__(
            self.DH_params_home,
            self.joint_types,
            self.mdh,
            self.joint_states_topic,
            self.base_link,
            self.target_link,

            func_post=self.func_post_scara)

        # Joint publishers
        self.pubs = [
            rospy.Publisher(
                f'/scara/joint{i}_position_controller/command',
                Float64, queue_size=1)
            for i in range(1, 4)
        ]
        self.joints = [1, 1, 1]

    def func_post_scara(self):
        # Post-test function (runs after every test)
        # moves the robot to a random configuration
        self.joints = [random.uniform(-1, 1) for i in range(3)]
        for i in range(3):
            self.pubs[i].publish(Float64(self.joints[i]))
        rospy.sleep(6)  # wait for the joints to reach targets


class UR5Test(RobotTest):

    # Params from modified URDF
    # DH_params_home = [
    #     [0, 0.089159, math.pi/2, 0],
    #     [-0.425, 0, 0, -math.pi/2],
    #     [-0.39225, 0, 0, 0],
    #     [0, 0.10915, math.pi/2, -math.pi/2],
    #     [0, 0.09465, -math.pi/2, 0],
    #     [0, 0, 0, 0]
    # ]
    DH_params_home = [
        [0,         0.089159,   0,          0],
        [0,         0,          math.pi/2,  0],
        [-0.425,    0,          0,          0],
        [-0.39225,  0.10915,    0,          0],
        [0,         0.09465,    math.pi/2,  0],
        [0,         0.0823,     -math.pi/2, 0]
    ]
    joint_types = ['theta'] * 6
    mdh = True
    joint_states_topic = '/joint_states'
    # base_link = '/shoulder_link'
    base_link = '/base_link_inertia'
    target_link = '/wrist_3_link'
    frames = ['shoulder_link', 'upper_arm_link', 'forearm_link',
              'wrist_1_link', 'wrist_2_link', 'wrist_3_link']
    fk_use_tf = False

    joint_limits = [(-0.75*math.pi, 0.75*math.pi),
                    (-0.75*math.pi, 0.75*math.pi),
                    (-0.75*math.pi, 0.75*math.pi),
                    (-0.75*math.pi, 0.75*math.pi),
                    (-0.75*math.pi, 0.75*math.pi),
                    (-0.75*math.pi, 0.75*math.pi),
                    (-0.75*math.pi, 0.75*math.pi)]

    def __init__(self):
        super().__init__(
            self.DH_params_home,
            self.joint_types,
            self.mdh,
            self.joint_states_topic,
            self.base_link,
            self.target_link,

            frames=self.frames,
            fk_use_tf=self.fk_use_tf,

            func_init=self.ws_func_init,
            # func_pre=self.func_pre_panda_vel,
            func_post=self.ws_func_post,
            # func_end=self.func_pre_panda_vel
        )

        # Joint velocity pub
        self.vel_pub = rospy.Publisher(
            "/joint_group_vel_controller/command",
            Float64MultiArray, queue_size=1
        )

        # Joints to be covered (workspace plotting)
        # Note: 1-based
        self.ws_joints = [2, 3, 4]

        self.joints = [1, 1, 1]

        # Communication with workspace_publisher
        self.wsp_trigger_pub = rospy.Publisher(
            '/workspace_publisher/trigger', Bool, queue_size=1)

    def process_joint_pos(self, joint_pos):
        """
        Preprocess joint positions if needed.
        For UR5, joints need to be reordered.
        """
        return list(reversed(joint_pos[:3])) + list(joint_pos[3:])

    def ws_publish_joint(self, i, val):
        """
        ***For Workspace plotting***
        Publishes [val] to joint [i]th in [self.ws_joints] list. (0-based)
        """
        vel = [0] * 6
        vel[self.ws_joints[i]-1] = val
        self.joint_vel_pub(vel)

    def ws_publish_all(self, vals):
        """
        ***For Workspace plotting***
        Publishes a list [vals] to selected ws joints.
        """
        vel = [0] * 6
        for i in range(len(self.ws_joints)):
            vel[self.ws_joints[i]-1] = vals[i]
        self.joint_vel_pub(vel)

    def ws_get_num_joints(self):
        """
        ***For Workspace plotting***
        Returns number of joints to plot workspace.
        """
        return len(self.ws_joints)

    def joint_vel_pub(self, joint_vel):
        msg = Float64MultiArray()
        msg.data = joint_vel
        self.vel_pub.publish(msg)

    def stop_joints(self):
        # stops the joints
        self.joint_vel_pub([0] * 6)

    def ws_func_init(self):
        # Move joints 2, 4, 6 to lower limit positions
        self.ws_publish_all([-1] * self.ws_get_num_joints())
        rospy.sleep(2)
        self.stop_joints()

    def ws_func_post(self):
        self.wsp_trigger_pub.publish(Bool(True))
