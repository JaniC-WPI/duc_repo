#!/usr/bin/env python3.8

import numpy as np
import cv2
import rospy
from std_msgs.msg import Bool, Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image as Img

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights_path = '/home/jc-merlab/Pictures/Data/trained_models/keypointsrcnn_planning_b2_e100_v4.pth'
model = torch.load(weights_path).to(device)
bridge = CvBridge()
i = 0
ros_img = None
kp = None

def visualize(img, bboxes, keypoints_list):
    # Draw bounding boxes
    for bbox in bboxes:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

    # Draw the keypoints
    for kp in keypoints_list:
        print(type(kp), kp)  # Debug print
        cv2.circle(img, (kp[0], kp[1]), 5, (0, 255, 0), -1)

    return img

previous_keypoints = []

def image_callback(msg):
    global ros_img, i, bridge, model, previous_keypoints

    ros_img = msg

    cv_img = bridge.imgmsg_to_cv2(ros_img, "bgr8")
    inf_img = Img.fromarray(cv_img)
    inf_img = F.to_tensor(inf_img).to(device)
    inf_img.unsqueeze_(0)
    inf_img = list(inf_img)
    
    with torch.no_grad():
        model.to(device)
        model.eval()
        output = model(inf_img)

    scores = output[0]['scores'].detach().cpu().numpy()
    high_scores_idxs = np.where(scores > 0.7)[0].tolist()
    post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()
    
    keypoints = []
    for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        keypoints.append(list(map(int, kps[0,0:2])))
    
    bboxes = [list(map(int, bbox.tolist())) for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()]

    print("keypoints", keypoints)

    previous_keypoints.append(keypoints)

    # Draw trajectory on the image using previous keypoints
    # for frame_kps in previous_keypoints:
    #     for idx in range(len(frame_kps) - 1):
    #         cv2.line(cv_img, tuple(frame_kps[idx]), tuple(frame_kps[idx + 1]), (0, 0, 255), 2)

    # Visualize current keypoints and bounding boxes
    img_with_kps = visualize(cv_img, bboxes, keypoints)

    # Save the image to a directory
    image_save_path = f"/home/jc-merlab/Pictures/panda_data/track_results/track_2d/{i:06}.png"
    cv2.imwrite(image_save_path, img_with_kps)
    
    video_writer.write(img_with_kps)  # Write the frame to the video

    i += 1

# Set up the video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('/home/jc-merlab/Pictures/panda_data/track_results/track_2d/trajectory.avi', fourcc, 20.0, (640, 480))  

if __name__ == '__main__':
    rospy.init_node('tracking_inference_node', anonymous=True)
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
    rospy.spin()