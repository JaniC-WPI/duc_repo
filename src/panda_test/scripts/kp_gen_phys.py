#!/usr/bin/env python3
"""
This version has factored out the robot-related code.
"""

import rospy
import tf
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_msgs.msg import Bool, Float64MultiArray
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
from cv_bridge import CvBridge
import json
import os
from Robot import RobotTest, PandaReal2D
# from visualize_kp import visualize
from datetime import datetime


# prefix to the image and json files names
int_stream = '000000'
folder = 9
# folder for main dataset
root_data_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/'


# homogenous tranformation from 4X1 translation and
def transform(tvec, quat):

    r = R.from_quat(quat).as_matrix()

    ht = np.array([[r[0][0], r[0][1], r[0][2], tvec[0]],
                   [r[1][0], r[1][1], r[1][2], tvec[1]],
                   [r[2][0], r[2][1], r[2][2], tvec[2]],
                   [0, 0, 0, 1]])

    return ht


class KpDetection():

    def __init__(self, robot: RobotTest,
                 save_dir: str = None,
                 iterations=None, sync=True, rate=10,
                 remove_dup=True,
                 screen_output=False,
                 no_kp_gen=False):
        """
        Initializes test object that runs for [iterations] tests.
        If [iterations] is not provided, run indefinitely.
        """
        self.robot = robot

        self.save_dir = save_dir

        # camera extrinsics
        self.camera_ext = None
        self.camera_ext_trans = None
        self.camera_ext_rot = None

        self.iterations = iterations  # number of tests

        self.ready = False  # Ready for next test
        self.world_coords_trig = False  # Determine if there is new data

        self.remove_dup = remove_dup
        self.screen_output = screen_output
        self.sync = sync
        self.rate = rate

        rospy.init_node('image_pix_gen', anonymous=True)

        rospy.Subscriber(self.robot.joint_states_topic, JointState,
                         self.world_coords_tf, queue_size=1)
        rospy.Subscriber(
            "/camera/color/image_raw", Image, self.get_image, queue_size=1)
        self.cam_intrinsics_sub = rospy.Subscriber(
            "/camera/color/camera_info", CameraInfo, self.camera_intrinsics,
            queue_size=1)
        self.tf_listener = tf.TransformListener()

        # Communication with workspace_publisher
        rospy.Subscriber('/workspace_publisher/movement_done', Bool,
                         self.wsp_movement_done, queue_size=1)
        rospy.Subscriber('/workspace_publisher/completed', Bool,
                         self.wsp_completed, queue_size=1)
        
        self.current_vel = []
        rospy.Subscriber('/workspace_publisher/current_vel', Float64MultiArray,
                         self.update_current_vel, queue_size=1)

        self.movement_done = True
        self.completed = False

        self.no_kp_gen = no_kp_gen

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
        
        rospy.sleep(2)

    def world_coords_tf(self, joints: JointState):
        """
        Calculates positions of frames in world frame using FK.
        """
        self.joint_angle = joints.position
        self.joint_vel = joints.velocity

        ht_list = self.robot.forward(self.joint_angle)

        tmp = np.transpose([0, 0, 0, 1])
        # Using local variable to prevent shared data bug
        world_coords = [np.dot(np.eye(4), tmp)]
        world_coords.extend([np.dot(ht, tmp) for ht in ht_list])
        self.world_coords = world_coords

        # Synchronize - unset flag to signal new data
        if self.world_coords_trig and self.ros_img is not None:
            self.world_coords_trig = False

    def kp_gen(self, flag, img: Image, id):
        """
        Saves image and json file containing keypoints.
        """
        # save the image with name being [id]
        cv_img = self.bridge.imgmsg_to_cv2(img, "bgr8")

        # process file name
        new_stream = int_stream[0:-len(str(id))]
        image_file = new_stream + str(id) + ".jpg"
        cv2.imwrite(os.path.join(self.save_dir, image_file), cv_img)
        rospy.loginfo(f'kp_gen(): Saved image {self.save_dir + image_file}')

        if not self.no_kp_gen:
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
                    ] for i in range(len(self.image_pix))
                    if not self.remove_dup or
                        (self.remove_dup and i not in self.robot.duplicated_joints)
                ],
                "keypoints": [
                    [
                        [
                            self.image_pix[i][0],
                            self.image_pix[i][1],
                            1
                        ]
                    ] for i in range(len(self.image_pix))
                    if not self.remove_dup or
                        (self.remove_dup and i not in self.robot.duplicated_joints)
                ],
            }

            # if len(data["bboxes"]) < self.robot.num_joints:
            #     rospy.logerr('Invalid length of [data]')

            # save [data] to json file
            json_obj = json.dumps(data, indent=4)
            filename = os.path.join(self.save_dir, new_stream + str(id)+".json")
            with open(filename, "w") as outfile:
                outfile.write(json_obj)
            rospy.loginfo(f'kp_gen(): Saved json {filename}')

            # Save the current velocity data to JSON file
            velocity_data = {
                "id": id,
                "velocity": self.current_vel
            }
            velocity_file_name = os.path.join(self.save_dir, f"{int_stream[:len(int_stream)-len(str(id))]}{id}_vel.json")
            with open(velocity_file_name, "w") as velocity_file:
                json.dump(velocity_data, velocity_file, indent=4)
            rospy.loginfo(f'kp_gen(): Saved velocity json {velocity_file_name}')

             # Save the current joint angles to JSON file
            # Save the specific joint angles to JSON file
            specific_joint_angles = [self.joint_angle[i] for i in [1, 3, 5]] 
            joint_angle_data = {
                "id": id,
                "joint_angles": specific_joint_angles
            }
            joint_angle_file_name = os.path.join(self.save_dir, f"{int_stream[:len(int_stream)-len(str(id))]}{id}_joint_angles.json")
            with open(joint_angle_file_name, "w") as joint_angle_file:
                json.dump(joint_angle_data, joint_angle_file, indent=4)
            rospy.loginfo(f'kp_gen(): Saved joint angles json {joint_angle_file_name}')

        # if self.screen_output:
        #     visualize(cv_img, data['keypoints'], data['bboxes'],
        #               int(1/self.rate*0.75*1000))  # wait time = 0.75 * 1/rate

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
        # if not self.no_kp_gen:
        #     (self.camera_ext_trans, self.camera_ext_rot) = \
        #         self.tf_listener.lookupTransform(
        #             '/camera_optical_link', self.robot.base_link, rospy.Time())
        #     self.camera_ext = transform(self.camera_ext_trans, self.camera_ext_rot)
        # for physical panda
        if not self.no_kp_gen:
            self.camera_ext_trans = [-0.15838935, 0.51719157,  1.74013216]#[-0.1406737, 0.51277635, 1.82787528]
            self.camera_ext_rot =  [0.66974827,  0.04262454, -0.03845967,  0.74036563] #[0.69402915, -0.17008132,  0.15281551,  0.68267365]
            self.camera_ext = transform(self.camera_ext_trans, self.camera_ext_rot)

    def image_pixels(self, camera_ext, world_coords):
        """
        Calculates pixel coordinates of the keypoints given extrinsics and
        keypoints' positions in world frame.
        """
        if not self.no_kp_gen:
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
        return None

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

    def update_current_vel(self, msg: Float64MultiArray):
        self.current_vel = msg.data

    def wsp_movement_done(self, msg: Bool):
        self.movement_done = msg.data

    def wsp_completed(self, msg: Bool):
        self.completed = msg.data

    def run(self):
        """
        Runs [self.iterations] tests
        """
        print("is run in kp_gen running!")
        # wait until camera intrinsics are retrieved
        while not self.no_kp_gen and self.camera_K is None:
            pass
        rospy.loginfo('camera_K initialized')

        while not self.no_kp_gen and self.world_coords is None:
            pass
        rospy.loginfo('world_coords initialized')

        iterations_str = self.iterations if self.iterations is not None \
            else 'inf'

        rate = rospy.Rate(self.rate)

        # Runs init function
        if self.robot.func_init is not None:
            rospy.loginfo('Running func_init()')
            self.robot.func_init()
            rospy.loginfo('func_init() completed')

        # for t in range(self.iterations):
        t = 0
        while not rospy.is_shutdown() and \
                ((self.iterations is None and not self.completed) or
                 (self.iterations is not None and t < self.iterations)):

            rospy.loginfo(f'Running test #{t+1}/{iterations_str}')

            # Runs [robot.func_pre] before each test
            if self.robot.func_pre is not None:
                rospy.loginfo('Running func_pre()')
                self.robot.func_pre()
                rospy.loginfo('func_pre() completed')

            # Wait until all robot movements done
            while self.sync and not self.movement_done:
                pass
            self.movement_done = False

            # Initialize new test
            self.ros_img = None
            self.world_coords_trig = False
            self.ready = True  # Ready to start new test
            # Wait until get new image
            while self.ros_img is None:
                pass
            # Wait until get new world_coord:
            while not self.no_kp_gen and self.world_coords_trig:
                pass

            rospy.loginfo('Generating keypoints')

            # Generates keypoints
            self.image_pix = self.image_pixels(
                self.camera_ext, self.world_coords)
            self.kp_gen(self.control_flag, self.ros_img, t)

            print(f'Current velocity: {self.current_vel}')

            # Runs [robot.func_post] after each test
            if self.robot.func_post is not None:
                rospy.loginfo('Running func_post()')
                self.robot.func_post()
                rospy.loginfo('func_post() completed')

            if not self.sync:
                rate.sleep()
            t += 1

        # Runs end function
        if self.robot.func_end is not None:
            rospy.loginfo('Running func_end()')
            self.robot.func_end()
            rospy.loginfo('func_end() completed')

        rospy.spin()


if __name__ == '__main__':
    # Create folder if not exists
    rospy.loginfo('Creating data folder...')
    timestamp = str(datetime.now()).split('.')[0].replace(':', '_').replace(' ', '_')
    save_path = os.path.join(root_data_dir, timestamp)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Run tests

    # KpDetection(PandaTest(), iterations=100).run()
    # KpDetection(PandaTest(), remove_dup=True).run()

    # robot = UR5Test()
    # kp = KpDetection(robot, sync=True)
    # robot.tf_listener = tf.TransformListener()
    # kp.run()
    KpDetection(PandaReal2D(), save_dir=save_path, rate=20, screen_output=False, sync=False, no_kp_gen=False).run()

    # KpDetection(ScaraTest(), iterations=100).run()
