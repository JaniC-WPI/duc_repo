import rospy
import random
import math
from std_msgs.msg import Float64, Bool
from DH import Kinematics
from sensor_msgs.msg import JointState


class RobotTest:
    """
    Collection of robot-specific parameters and test functions.
    """

    def __init__(self, DH_params_home, joint_types, mdh,
                 joint_states_topic, base_link, target_link,
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

        func_init: Runs once before all tests
        func_pre: Runs before every test
        func_post: Runs after every test
        func_end: Runs once after all tests
        """
        self.kinematics = Kinematics(DH_params_home, joint_types, mdh=mdh)
        self.num_joints = len(DH_params_home)

        self.func_init = func_init  # Runs once before tests
        self.func_pre = func_pre  # Runs before every test
        self.func_post = func_post  # Runs after every test
        self.func_end = func_end  # Runs once after all tests

    def process_joint_pos(self, joint_pos):
        """
        Preprocess joint positions if needed.
        """
        return joint_pos


class PandaTest(RobotTest):

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

    def __init__(self):
        super().__init__(
            self.DH_params_home,
            self.joint_types,
            self.mdh,
            self.joint_states_topic,
            self.base_link,
            self.target_link,

            func_init=self.func_init_wsp,
            # func_pre=self.func_pre_panda_vel,
            func_post=self.func_post_wsp,
            # func_end=self.func_pre_panda_vel
        )

        # Joint publishers
        self.pubs = [
            rospy.Publisher(
                f'/panda/joint{i*2}_velocity_controller/command',
                Float64, queue_size=1)
            for i in range(1, 4)
        ]
        self.joints = [1, 1, 1]

        self.joint_limits = [(-2.8973, 2.8973),
                             (-1.7628, 1.7628),
                             (-2.8973, 2.8973),
                             (-3.0718, -0.0698),
                             (-2.8973, 2.8973),
                             (-0.0175, 3.7525),
                             (-2.8973, 2.8973)]

        # Communication with workspace_publisher
        self.wsp_trigger_pub = rospy.Publisher(
            '/workspace_publisher/trigger', Bool, queue_size=1)

    def process_joint_pos(self, joint_pos):
        return joint_pos[2:]

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
        for i in range(len(joint_vel)):
            self.pubs[i].publish(Float64(joint_vel[i]))

    def stop_joints(self):
        # stops the joints
        self.joint_vel_pub([0, 0, 0])

    def func_init_wsp(self):
        # Move joints 2, 4, 6 to lower limit positions
        self.joint_vel_pub([-1, -1, -1])
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
