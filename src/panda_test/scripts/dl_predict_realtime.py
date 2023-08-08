#!/usr/bin/env python3

"""
Predicts using R-CNN model in real time.
"""

import os
import rospy
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import cv2
from cv_bridge import CvBridge
import torch
import torchvision
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt


int_stream = '0000000'


def preprocess_img(img):
    """
    Converts image to tensor to put into Model.
    """
    img_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return F.to_tensor(img_tmp)


def visualize(image, bboxes, keypoints, image_original=None, bboxes_original=None, keypoints_original=None):
    fontsize = 18

    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)

    for idx, kps in enumerate(keypoints):
        for kp in kps:
            image = cv2.circle(image.copy(), tuple(kp), 2, (255,0,0), 10)

    if image_original is None and keypoints_original is None:
        # plt.figure(figsize=(40,40))
        # plt.imshow(image)

        return image

    else:
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)

        print(keypoints_original)
        for idx, kps in enumerate(keypoints_original):
            print(idx)
            print(kps)
            for kp in kps:
                print(kp)
                image_original = cv2.circle(image_original, tuple(kp), 5, (255,0,0), 2)

        f, ax = plt.subplots(1, 2, figsize=(40, 20))

        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)

        return None


class KpDLVisualizer():
    """
    Visualize prediction from KP detection network in real time.
    """

    def __init__(self, model_path, im_plt, save_img=False, img_dir_path=None):
        """
        model_path: str: path to keypoint detection model
        im_plt: unused
        save_img: save the prediction to file
        img_dir_path: if [save_img] is True, this will be the path to save images
        """
        self.model_path = model_path
        self.save_img = save_img
        self.img_dir_path = img_dir_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.load(self.model_path).to(self.device)
        self.model.eval()

        # ROS Node
        rospy.init_node('kp_dl_visualizer', anonymous=True)

        # Subscribes to Camera topics
        rospy.Subscriber(
            "/camera/color/image_raw", Image, self.get_image, queue_size=1)
        self.cam_intrinsics_sub = rospy.Subscriber(
            "/camera/color/camera_info", CameraInfo, self.camera_intrinsics,
            queue_size=1)

        self.bridge = CvBridge()

        self.i = 1  # Counter for images

        # Matplotlib figures
        # self.im_plt = im_plt

        rospy.sleep(1)

    def visualize(self, img):
        img_tensor = preprocess_img(img).to(self.device)
        with torch.no_grad():
            outputs = self.model([img_tensor])
            img_tensor_tup = (
                img_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255
            ).astype(np.uint8)
            scores = outputs[0]['scores'].detach().cpu().numpy()
            # Indexes of boxes with scores > 0.7
            high_scores_idxs = np.where(scores > 0.7)[0].tolist()
            # Indexes of boxes left after applying NMS (iou_threshold=0.3)
            post_nms_idxs = torchvision.ops.nms(
                outputs[0]['boxes'][high_scores_idxs],
                outputs[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()

            keypoints = []
            for kps in outputs[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                keypoints.append([list(map(int, kp[:2])) for kp in kps])

            bboxes = []
            for bbox in outputs[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
                bboxes.append(list(map(int, bbox.tolist())))
            img = visualize(img_tensor_tup, bboxes, keypoints)

            if self.save_img:
                new_stream = int_stream[0:-len(str(self.i))]
                image_file = new_stream + str(self.i) + ".rgb.jpg"
                cv2.imwrite(self.img_dir_path + image_file, img)
                rospy.loginfo(f'kp_gen(): Saved image {file_path + image_file}')
                self.i += 1
            else:
                cv2.imshow('Camera', img)
                cv2.waitKey(30)  # 30 milisec
            # self.im_plt.set_data(img)

    def get_image(self, image: Image):
        """
        Get an image for processing when [self.ready] flag is set.
        Camera extrinsics matrix is also calculated to ensure consistency.
        """
        cv_img = self.bridge.imgmsg_to_cv2(image, "bgr8")
        self.visualize(cv_img)

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
                '/camera_optical_link', self.robot.base_link, rospy.Time())
        # self.camera_ext = transform(self.camera_ext_trans, self.camera_ext_rot)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    model_path = '/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/trained_models/keypointsrcnn_weights_ld_b1_e25_v9.pth'

    # DL prediction image data
    folder = 9
    pred = 1
    file_path = f'/home/user/Workspace/WPI/Summer2023/ws/duc_repo/src/panda_test/data/dl_prediction_result/{folder}/{pred}/'
    # Create folder if not exists
    if not os.path.exists(file_path):
        rospy.logwarn('Data folder not found. Creating one...')
        os.mkdir(file_path)

    # Check GPU memory
    rospy.loginfo('Checking GPU memory')
    t = torch.cuda.get_device_properties(0).total_memory
    print(t)
    torch.cuda.empty_cache()

    r = torch.cuda.memory_reserved(0)
    print(r)
    a = torch.cuda.memory_allocated(0)
    print(a)
    rospy.loginfo('GPU memory check completed. Running node...')

    # im_plt = plt.imshow(np.zeros((480, 640), dtype=np.int8))
    im_plt = None
    # plt.ion()

    KpDLVisualizer(model_path=model_path, im_plt=im_plt, save_img=False, img_dir_path=file_path).run()

    # plt.ioff()
    # plt.show()
