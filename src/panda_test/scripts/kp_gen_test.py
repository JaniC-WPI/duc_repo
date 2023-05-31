#!/usr/bin/env python3

import rospy
import tf
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_msgs.msg import Float64
from scipy.spatial.transform import Rotation as R
import math
import numpy as np
import cv2
from cv_bridge import CvBridge
import json
import random
from DH import Kinematics


# Panda
DH_params_home = [
    [0,      0.333,             0,   0],
    [0,          0,  -(math.pi/2),   0],
    [0,      0.316,   (math.pi/2),   0],
    [0.0825,     0,   (math.pi/2),   0],
    [-0.0825, 0.384,  -(math.pi/2),  0],
    [0,          0,   (math.pi/2),   0],
    [0.088,      0,   (math.pi/2),   0]]
    # [0.1,      0.1070,   0,          0]]
joint_types = ['t', 't', 't', 't', 't', 't', 't']
mdh = True
joint_states_topic = '/panda/joint_states'
base_link = '/panda_link0'
target_link = '/panda_link7'

# SCARA
# DH_params_home = [
#     [0,       0.05, 0,        0],
#     [0.425,   0.45, 0,        0],
#     [0.345,  -0.06, 3.14,  0]]
# joint_types = ['t', 't', 'd']
# mdh = False
# joint_states_topic = '/scara/joint_states'
# base_link = '/base_link'
# target_link = '/link3'

# prefix to the image and json files names
int_stream = '000000'
folder = 4
# folder for main dataset
file_path = f'/home/user/Workspace/WPI/Summer2023/ws/src/panda_test/data/kp_test_images/{folder}/'


# homogenous tranformation from 4X1 translation and
def transform(tvec, quat):

    r = R.from_quat(quat).as_matrix()

    ht = np.array([[r[0][0], r[0][1], r[0][2], tvec[0]],
                   [r[1][0], r[1][1], r[1][2], tvec[1]],
                   [r[2][0], r[2][1], r[2][2], tvec[2]],
                   [0, 0, 0, 1]])

    return ht


def func_post_panda_pos():
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


def func_post_panda_vel():
    # Post-test function (runs after every test)
    # moves the robot to a random configuration with joints 2, 4, 6

    # Using static variables for storing previous joint values
    if not hasattr(func_post_panda_vel, "pubs"):
        func_post_panda_vel.pubs = [
            rospy.Publisher(
                f'/panda/joint{i*2}_velocity_controller/command',
                Float64, queue_size=1)
            for i in range(1, 4)
        ]

    if not hasattr(func_post_panda_vel, "joints"):
        func_post_panda_vel.joints = [1, 1, -1]

    for i in range(3):
        # New velocity has opposite sign with previous value to prevent
        # reaching joint limits
        func_post_panda_vel.joints[i] = \
            math.copysign(random.uniform(-1, 1),
                          -func_post_panda_vel.joints[i])
        func_post_panda_vel.pubs[i].publish(
            Float64(func_post_panda_vel.joints[i]))


def func_pre_panda_vel():
    # Pre-test function (runs before every test)
    # stops the joints
    pub2 = rospy.Publisher(
            "/panda/joint2_velocity_controller/command", Float64, queue_size=1)
    pub4 = rospy.Publisher(
            "/panda/joint4_velocity_controller/command", Float64, queue_size=1)
    pub6 = rospy.Publisher(
            "/panda/joint6_velocity_controller/command", Float64, queue_size=1)
    pub2.publish(Float64(0))
    pub4.publish(Float64(0))
    pub6.publish(Float64(0))


def func_post_scara():
    # Post-test function (runs after every test)
    # moves the robot to a random configuration
    pubs = [
        rospy.Publisher(
            f'/scara/joint{i}_position_controller/command',
            Float64, queue_size=1)
        for i in range(1, 4)
    ]
    joints = [random.uniform(-1, 1) for i in range(3)]
    for i in range(3):
        pubs[i].publish(Float64(joints[i]))
    rospy.sleep(6)  # wait for the joints to reach targets


class KpDetection():

    def __init__(self, iterations=None, func_init=None, func_pre=None,
                 func_post=None, func_end=None):
        rospy.init_node('image_pix_gen', anonymous=True)

        rospy.Subscriber(joint_states_topic, JointState,
                         self.world_coords_tf, queue_size=1)
        rospy.Subscriber(
            "/camera/color/image_raw", Image, self.get_image, queue_size=1)
        self.cam_intrinsics_sub = rospy.Subscriber(
            "/camera/color/camera_info", CameraInfo, self.camera_intrinsics,
            queue_size=1)
        self.tf_listener = tf.TransformListener()

        self.kinematics = Kinematics(DH_params_home, joint_types, mdh=mdh)
        self.num_joints = len(DH_params_home)
        # camera extrinsics
        self.camera_ext = None
        self.camera_ext_trans = None
        self.camera_ext_rot = None

        self.iterations = iterations  # number of tests
        self.func_init = func_init  # Runs once before tests
        self.func_pre = func_pre  # Runs before every test
        self.func_post = func_post  # Runs after every test
        self.func_end = func_end  # Runs once after all tests

        self.ready = False  # Ready for next test
        self.world_coords_trig = False  # Determine if there is new data

        # From original kp_detection.py
        self.bridge = CvBridge()
        self.camera_K = None  # camera intrinsics
        self.world_coords = None  # world coordinates of frames
        self.image_pix = None  # frame pixels of the current image
        self.joint_vel = None  # joints' velocities
        self.joint_angle = None  # joints' angles
        self.control_flag = None
        self.ros_img = None  # image in processing
        self.status = None

    def world_coords_tf(self, joints: JointState):
        """
        Calculates positions of frames in world frame using FK.
        """
        self.joint_angle = joints.position
        self.joint_vel = joints.velocity

        # panda
        ht_list = self.kinematics.forward(self.joint_angle[2:])
        # scara
        # ht_list = self.kinematics.forward(self.joint_angle)

        tmp = np.transpose([0, 0, 0, 1])
        world_coords = [np.dot(np.eye(4), tmp)]
        world_coords.extend([np.dot(ht, tmp) for ht in ht_list])
        self.world_coords = world_coords

        # Synchronize - unset flag to signal new data
        if self.world_coords_trig and self.ros_img is not None:
            self.world_coords_trig = False

        if len(self.world_coords) < self.num_joints:
            rospy.logerr('Invalid length of [self.world_coords]')

    def kp_gen(self, flag, img: Image, id):
        """
        Saves image and json file containing keypoints.
        """
        # save the image with name being [id]
        cv_img = self.bridge.imgmsg_to_cv2(img, "rgb8")

        # process file name
        new_stream = int_stream[0:-len(str(id))]
        image_file = new_stream + str(id) + ".rgb.jpg"
        cv2.imwrite(file_path + image_file, cv_img)
        rospy.loginfo(f'kp_gen(): Saved image {file_path + image_file}')

        # keypoint json
        data = {
            "id": id,
            "image_rgb": image_file,
            "bboxes": [
                [
                    self.image_pix[i][0]-10,
                    self.image_pix[i][1]-10,
                    self.image_pix[i][0]+10,
                    self.image_pix[i][1]+10
                ]
                for i in range(len(self.image_pix))
            ],
            "keypoints": [
                [
                    [
                        self.image_pix[i][0],
                        self.image_pix[i][1],
                        1
                    ]
                ]
                for i in range(len(self.image_pix))
            ],
        }

        # save [data] to json file
        json_obj = json.dumps(data, indent=4)
        filename = file_path + new_stream + str(id)+".json"
        with open(filename, "w") as outfile:
            outfile.write(json_obj)
        rospy.loginfo(f'kp_gen(): Saved json {filename}')

    def camera_intrinsics(self, camera_info: CameraInfo):
        """
        Get camera intrinsics matrix.
        """
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]
        self.camera_K = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

        # Unsubscribe after getting the intrinsics
        rospy.loginfo('Updated camera intrinsics. Unsubscribing...')
        self.cam_intrinsics_sub.unregister()

    def camera_extrinsics(self):
        """
        Updates camera extrinsics matrices.
        """
        (self.camera_ext_trans, self.camera_ext_rot) = \
            self.tf_listener.lookupTransform(
                '/camera_optical_link', base_link, rospy.Time())
        self.camera_ext = transform(self.camera_ext_trans, self.camera_ext_rot)

    def image_pixels(self, camera_ext, world_coords):
        """
        Calculates pixel coordinates of the keypoints given extrinsics and
        keypoints' positions in world frame.
        """
        proj_model = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
        # make intrinsics matrix 4x4
        camera_K_4 = np.dot(self.camera_K, proj_model)
        # raw (unscaled) keypoint pixels
        image_pixs = [
            np.dot(camera_K_4, np.dot(camera_ext, world_coords[i]))
            for i in range(len(world_coords))]
        # scale the pixel coordinates
        img_pixels = [
            (
                image_pixs[i][0]/image_pixs[i][2],  # u
                image_pixs[i][1]/image_pixs[i][2],  # v
            ) for i in range(len(world_coords))]
        return img_pixels

    def get_image(self, image: Image):
        """
        Get an image for processing when [self.ready] flag is set.
        Camera extrinsics matrix is also calculated to ensure consistency.
        """
        if self.ready:
            self.ready = False
            self.world_coords_trig = True
            self.camera_extrinsics()
            self.ros_img = image

    def run(self):
        """
        Runs [self.iterations] tests
        """

        # wait until camera intrinsics are retrieved
        while self.camera_K is None:
            pass
        rospy.loginfo('camera_K initialized')

        while self.world_coords is None:
            pass
        rospy.loginfo('world_coords initialized')

        iterations_str = self.iterations if self.iterations is not None \
            else 'inf'

        # Runs init function
        if self.func_init is not None:
            rospy.loginfo('Running func_init()')
            self.func_init()
            rospy.loginfo('func_init() completed')

        # for t in range(self.iterations):
        t = 0
        rate = rospy.Rate(5)
        while not rospy.is_shutdown() and \
                (self.iterations is None or t < self.iterations):

            rospy.loginfo(f'Running test #{t+1}/{iterations_str}')

            # Runs [self.func_pre] before each test
            if self.func_pre is not None:
                rospy.loginfo('Running func_pre()')
                self.func_pre()
                rospy.loginfo('func_pre() completed')

            # Initialize new test
            self.ros_img = None
            self.world_coords_trig = False
            self.ready = True  # Ready to start new test
            # Wait until get new image
            while self.ros_img is None:
                pass
            # Wait until get new world_coord:
            while self.world_coords_trig:
                pass

            rospy.loginfo('Generating keypoints')

            # Generates keypoints
            self.image_pix = self.image_pixels(
                    self.camera_ext, self.world_coords)
            self.kp_gen(self.control_flag, self.ros_img, t)

            # Runs [self.func_post] after each test
            if self.func_post is not None:
                rospy.loginfo('Running func_post()')
                self.func_post()
                rospy.loginfo('func_post() completed')

            rate.sleep()
            t += 1

        # Runs end function
        if self.func_end is not None:
            rospy.loginfo('Running func_end()')
            self.func_end()
            rospy.loginfo('func_end() completed')

        rospy.spin()


if __name__ == '__main__':
    KpDetection(iterations=100, func_pre=func_pre_panda_vel,
                func_post=func_post_panda_vel,
                func_end=func_pre_panda_vel) \
        .run()
    # KpDetection(10, func_post=func_post_scara).run()
