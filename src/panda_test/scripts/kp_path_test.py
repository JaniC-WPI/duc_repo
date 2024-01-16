#!/usr/bin/env python3.8

import rospy
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_msgs.msg import Bool, Float64MultiArray, Float64
import numpy as np
import cv2
from cv_bridge import CvBridge
import torch
import torchvision
from torchvision.transforms import functional as F
import json
import os
from PIL import Image as Img
import os
from vel_regression import VelRegModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights_path = '/home/jc-merlab/Pictures/Data/trained_models/keypointsrcnn_weights_sim_b1_e25_v0.pth'
# weights_path = rospy.get_param('vsbot/deeplearning/weights_path')
kp_model = torch.load(weights_path).to(device)
bridge = CvBridge()
model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_vel_b128_e200_v0.pth'
vel_model = VelRegModel(input_size=18)
vel_model.load_state_dict(torch.load(model_path))
predicted_velocity = torch.zeros(3, dtype=torch.float32)

def kp_detection(ros_img):
    global predicted_velocity

    if ros_img is not None:
        cv_img = bridge.imgmsg_to_cv2(ros_img, "bgr8")

        inf_img = Img.fromarray(cv_img)

        inf_img = F.to_tensor(inf_img).to(device)
        inf_img.unsqueeze_(0)
        inf_img = list(inf_img)
        with torch.no_grad():
            kp_model.to(device)
            kp_model.eval()
            output = kp_model(inf_img)
        inf_img = (inf_img[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0].tolist() # Indexes of boxes with scores > 0.7
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], \
            output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()
        
        keypoints = []
        for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            keypoints.append(list(map(int, kps[0,0:3])))
        
        labels = []
        for label in output[0]['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            labels.append(label)

        keypoints_ = [x for _,x in sorted(zip(labels,keypoints))]

        print(keypoints_)

        start_kp = keypoints_
        next_kp = [[258, 367, 1], [257, 282, 1], [253, 203, 1], [274, 203, 1], [334, 123, 1], [350, 135, 1]]

        # Flatten and concatenate start_kp and next_kp
        start_kp_flat = [kp for sublist in start_kp for kp in sublist]
        next_kp_flat = [kp for sublist in next_kp for kp in sublist]
        input_tensor = torch.tensor(start_kp_flat + next_kp_flat, dtype=torch.float32)

        # Reshape the input tensor to match the model's expected input shape ([1, input_size])
        input_tensor = input_tensor.view(1, -1)
        # Reshape the input tensor and split it into start_kp and next_kp
        input_tensor = input_tensor.view(1, -1)
        split_size = len(start_kp_flat)
        start_kp_tensor, next_kp_tensor = torch.split(input_tensor, [split_size, split_size], dim=1)


        # Make prediction
        vel_model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            pred_vel = vel_model(start_kp_tensor, next_kp_tensor)
            predicted_velocity = pred_vel.squeeze()

        # Visualize keypoints on the image
        for kp in start_kp:
            cv2.circle(cv_img, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)  # Green for start_kp
        for kp in next_kp:
            cv2.circle(cv_img, (int(kp[0]), int(kp[1])), 5, (255, 0, 0), -1)  # Blue for next_kp

        # Save the image
        img_save_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/vel_reg_sim_test/image.jpg'  # Update with your path
        cv2.imwrite(img_save_path, cv_img)

        
    print(predicted_velocity)     

def main():
    global predicted_velocity

    rospy.init_node('velocity_gen_test')

    vel_pub_2 = rospy.Publisher('/panda/joint2_velocity_controller/command', Float64, queue_size=1)
    vel_pub_4 = rospy.Publisher('/panda/joint4_velocity_controller/command', Float64, queue_size=1)
    vel_pub_6 = rospy.Publisher('/panda/joint6_velocity_controller/command', Float64, queue_size=1)

    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, kp_detection, queue_size=1)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        # Convert predicted_velocity to list here, within the loop
        velocity = predicted_velocity.tolist() if predicted_velocity.numel() > 0 else [0, 0, 0]

        print(velocity)

        jvel_2 = Float64()
        jvel_4 = Float64()
        jvel_6 = Float64()

        jvel_2.data = velocity[0]
        jvel_4.data = velocity[1]
        jvel_6.data = velocity[2]

        vel_pub_2.publish(jvel_2)
        vel_pub_4.publish(jvel_4)
        vel_pub_6.publish(jvel_6)

        rate.sleep()

    rospy.spin()

if __name__ == '__main__':
    main()











