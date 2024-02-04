#!/usr/bin/env python3.8

import numpy as np
import cv2
import rospy
from std_msgs.msg import Int32, Bool, Float64MultiArray
from sensor_msgs.msg import Image, CameraInfo, JointState
# from forward import Kinematics
from cv_bridge import CvBridge
import torch
import torchvision
from torchvision.transforms import functional as F
import os
from PIL import Image as Img
from datetime import datetime
from os.path import expanduser
from sklearn.neighbors import KDTree
import json
import networkx as nx
# from utils import DataPrePro
from panda_test.srv import dl_plan_sim_img, dl_plan_sim_imgResponse
# from visualization import visualize

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# weights_path = '/home/jc-merlab/Pictures/Data/trained_models/keypointsrcnn_weights_ld_b3_e25.pth'
weights_path = rospy.get_param('vsbot/deeplearning/weights_path')
model = torch.load(weights_path).to(device)
bridge = CvBridge()
i = 0
ros_img = None
kp = None
status_pub = rospy.Publisher("vsbot/status", Int32, queue_size=1)
final_goal =  rospy.get_param('dl_controller/goal_features')

SAFE_DISTANCE = 20  # Safe distance from the obstacle

# Load keypoints from JSON files in a given directory
def load_keypoints_from_json(directory):
    configurations = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                # Convert keypoints to integers
                keypoints = [np.array(point[0][:2], dtype=int) for point in data['keypoints']]  # Extracting x, y coordinates
                configurations.append(np.array(keypoints))
    # print(configurations)
    return configurations

def load_and_sample_configurations(directory, num_samples):
    # Load configurations from JSON files
    configurations = load_keypoints_from_json(directory)

    # If there are more configurations than needed, sample a subset
    if len(configurations) > num_samples:
        sampled_indices = np.random.choice(len(configurations), size=num_samples, replace=False)
        sampled_configurations = [configurations[i] for i in sampled_indices]
    else:
        sampled_configurations = configurations

    return sampled_configurations

def is_collision_free_line_check(configuration, obstacle_center, safe_distance):
    # Define a function to check for collision along a line segment
    def is_line_colliding(p1, p2, obstacle_center, safe_distance):
        # Check if the line segment is effectively a point
        if np.allclose(p1, p2):
            return np.linalg.norm(np.array(p1) - np.array(obstacle_center)) < safe_distance

        # Check for collision at regular intervals along the line
        num_checks = int(np.linalg.norm(np.array(p2) - np.array(p1)) / safe_distance)
        if num_checks == 0:
            # Avoid division by zero if points are extremely close
            return False

        for i in range(num_checks + 1):
            point = np.array(p1) + (np.array(p2) - np.array(p1)) * i / num_checks
            if np.linalg.norm(point - np.array(obstacle_center)) < safe_distance:
                return True
        return False

    for i in range(len(configuration) - 1):
        if is_line_colliding(configuration[i], configuration[i + 1], obstacle_center, safe_distance):
            return False
    return True


def build_roadmap_kd_tree(configurations, start_config, goal_config, k, obstacle_center, safe_distance):
    G = nx.Graph()
    all_configs = [start_config, goal_config] + configurations
    all_configs_np = np.array([c.flatten() for c in all_configs])

    # Create KDTree for efficient nearest neighbor search
    tree = KDTree(all_configs_np)

    for i, config in enumerate(all_configs):
         # Reshape the config to be 2D array for KDTree query
        config_reshaped = config.flatten().reshape(1, -1)
        # Query the k nearest neighbors
        distances, indices = tree.query(config_reshaped, k=k+1)  # k+1 because the query point itself is included

        for j in range(1, k+1):  # Skip the first index (itself)
            neighbor_config = all_configs[indices[0][j]]
            if is_collision_free_line_check(np.vstack([config, neighbor_config]), obstacle_center, safe_distance):
                G.add_edge(tuple(map(tuple, config)), tuple(map(tuple, neighbor_config)), weight=distances[0][j])

    return G

def find_path_prm(graph, start_config, goal_config):
    # Convert configurations to tuple for graph compatibility
    start = tuple(map(tuple, start_config))
    goal = tuple(map(tuple, goal_config))

    try:
        path = nx.astar_path(graph, start, goal)
        return path
    except nx.NetworkXNoPath:
        return None

# Detect a red ball in an image
def detect_red_ball(image):
    # image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Red color might be in two ranges
    lower_red_1 = np.array([0, 50, 50])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 50, 50])
    upper_red_2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask = mask1 + mask2  # Combine masks
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:        
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)  # Green circle with thickness of 2
        # Draw the center of the circle
        center_radius = 5  # Radius of the center point
        cv2.circle(image, (int(x), int(y)), center_radius, (255, 0, 0), -1)  # Blue filled circle for the center

        # Convert the modified image back to a ROS message and publish it
        modified_img_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
        # image_pub.publish(modified_img_msg)

        return (int(x), int(y), int(radius))
    return None

def image_callback(msg):
    global ros_img

    ros_img = msg

def dl_plan_image_service(img):
    global bridge, model, kp, ros_img

    print("svc ros image", type(ros_img))
    print("is keypoint service getting called")

    if ros_img is not None:
        status_msg = Int32()
        status_msg.data = 1
        status_pub.publish(status_msg)
        cv_img = bridge.imgmsg_to_cv2(ros_img, "bgr8")

        inf_img = Img.fromarray(cv_img)

        inf_img = F.to_tensor(inf_img).to(device)
        inf_img.unsqueeze_(0)
        inf_img = list(inf_img)
        with torch.no_grad():
            model.to(device)
            model.eval()
            output = model(inf_img)
        inf_img = (inf_img[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0].tolist() # Indexes of boxes with scores > 0.7
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], \
            output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)
        # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
        # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
        # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes
        keypoints = []
        key_points = []
        for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            keypoints.append(list(map(int, kps[0,0:2])))
            # for kp in kps:
                # print(kp)
            key_points.append([list(map(int, kp[:2])) for kp in kps])
        # print(np.array(keypoints).shape)                
        # if len(keypoints) == 6:
        #     keypoints.pop(2)

        labels = []
        for label in output[0]['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            labels.append(label)

        keypoints_ = [x for _,x in sorted(zip(labels,keypoints))]

        print(keypoints_)
            
        # print(keypoints)
        print("no of keypoints", len(keypoints))
        bboxes = []
        for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            bboxes.append(list(map(int, bbox.tolist())))
        
        # print("key points", keypoints)
        # img = visualize(inf_img, bboxes, key_points)
        # cv2.imwrite("/home/jc-merlab/Pictures/Data/video_results_live_b1e25_v2/out_image_" + str(i) + ".jpg", img)

        # uncomment the next line for 4 feature points
        # indices = [0,1,2,3,4,5,6,8]
        # uncomment the next line for 3 feature points
        indices = [0,1,2,3,4,5]
        keypoints_ = [keypoints_[i] for i in indices]

        print("keypoints")

        kp_x = []
        kp_y = []
        for i in range(len(keypoints_)):
            x = np.int64(keypoints_[i][0])
            y = np.int64(keypoints_[i][1])
            kp_x.append(x)
            kp_y.append(y)

        kp = []
        
        # Uncomment the next block for 3 features
        # for i in range(len(kp_x)-2):
        #    kp.append(kp_x[i+2]) 
        #    kp.append(kp_y[i+2])

        for i in range(len(kp_x)):
           kp.append(kp_x[i]) 
           kp.append(kp_y[i])



        # Uncomment the next block for 4 features
        # for i in range(len(kp_x)-3):
        #    kp.append(kp_x[i+3]) 
        #    kp.append(kp_y[i+3])

        kp_resp = Float64MultiArray()
        kp_resp.data = kp

        print("keypoints", kp_resp.data)
        print("type keypoints", type(kp_resp.data))

    # i = i+1
        return dl_plan_sim_imgResponse(ros_img, kp_resp)

    else:
        print("no image spawned")

  
def main():
    # Initialize the node
    rospy.init_node('goal_inference_gen')
    print("is main getting called")

    # Declaring the keypoints service
    kp_service = rospy.Service("franka_kp_dl_service", dl_plan_sim_img, dl_plan_image_service)
    # subscriber for rgb image to detect markers
    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, image_callback, queue_size=1)

    rospy.spin()

if __name__=='__main__':
    main()
