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
import os
from PIL import Image as Img
from att_unet import UNet  # Importing UNet model
from panda_test.srv import dl_sim_img, dl_sim_imgResponse
from kalmanfilter import KalmanFilter

# Device Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {device}")

# Load Pre-Trained Models
weights_path = rospy.get_param('vsbot/deeplearning/weights_path')
no_of_features = rospy.get_param('vsbot/shape_control/no_of_features')

# Load Keypoint Model
kp_model = torch.load(weights_path).to(device)

# Load Inpainting Model (UNet)
inpaint_path = rospy.get_param('vsbot/deeplearning/inpaint_path')  # UNet path from ROS param
inpaint_model = UNet().to(device)
inpaint_model.load_state_dict(torch.load(inpaint_path, map_location=device))
inpaint_model.eval()

# Initialize Kalman Filters for each keypoint
kalman_filters = [KalmanFilter(100, 7, 7, 7, 7) for _ in range(5)]

class VideoInference:
    def __init__(self):
        self.bridge = CvBridge()
        self.cv_image = None
        self.ros_img = None
        self.i = 0
        self.executed = False
        self.corrected_x = None

        # ROS Publishers & Subscribers
        self.flag_pub = rospy.Publisher("/franka/control_flag", Bool, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1)
        self.kp_service = rospy.Service("franka_kp_dl_service", dl_sim_img, self.dl_sim_image_service)

    def preprocess_image(self, img):
        """ Convert ROS image to PyTorch tensor for UNet inpainting. """
        # image = Img.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        image = Img.fromarray(img)
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((480, 640)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.2096, 0.2251, 0.2098], std=[0.2064, 0.1921, 0.2066]
            )
        ])
        return preprocess(image).unsqueeze(0).to(device)

    # def inpaint_image(self, img):
    #     """ Passes image through UNet inpainting model. """
    #     input_tensor = self.preprocess_image(img)
    #     with torch.no_grad():
    #         inpainted_image = inpaint_model(input_tensor)
    #     inpainted_image = self.denormalize(inpainted_image)
    #     return self.tensor_to_cv2(inpainted_image)

    def inpaint_image(self, img):
        """ Passes image through UNet inpainting model. """

        if img is None:
            rospy.logerr("Received NoneType image in inpaint_image function.")
            return None

        rospy.loginfo("Preprocessing image for inpainting...")

        input_tensor = self.preprocess_image(img)

        if input_tensor is None:
            rospy.logerr("Failed to preprocess image for inpainting.")
            return None

        rospy.loginfo("Passing image through UNet inpainting model...")

        with torch.no_grad():
            inpainted_image = inpaint_model(input_tensor)

        if inpainted_image is None:
            rospy.logerr("UNet model returned None for inpainted image.")
            return None

        inpainted_image = self.denormalize(inpainted_image)

        if inpainted_image is None:
            rospy.logerr("Denormalization failed.")
            return None

        # Convert to OpenCV format
        inpainted_cv2 = self.tensor_to_cv2(inpainted_image)

        if inpainted_cv2 is None:
            rospy.logerr("Conversion to OpenCV format failed.")
            return None

        debug_inpaint_path = "/home/jc-merlab/debug_inpainted.jpg"
        cv2.imwrite(debug_inpaint_path, inpainted_cv2)
        rospy.loginfo(f"Saved inpainted image to {debug_inpaint_path}")

        return inpainted_cv2

    def denormalize(self, tensor):
        """ Convert normalized tensor back to original pixel values. """
        means = torch.tensor([0.2096, 0.2251, 0.2098]).view(-1, 1, 1).to(device)
        stds = torch.tensor([0.2064, 0.1921, 0.2066]).view(-1, 1, 1).to(device)
        return (tensor * stds + means).clamp(0, 1)

    def tensor_to_cv2(self, tensor):
        """ Convert PyTorch tensor to OpenCV image format. """
        img_np = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    def image_callback(self, msg):
        """ ROS Callback: Receives image and sets the flag for processing. """
        self.ros_img = msg
        if self.ros_img is not None:
            self.flag_pub.publish(True)

    def dl_sim_image_service(self, img):
        """ ROS Service: Processes the received image, applies inpainting, and predicts keypoints. """
        # Convert ROS image to OpenCV
        self.cv_img = self.bridge.imgmsg_to_cv2(self.ros_img, "bgr8")

        # Step 1: **Pass Image Through UNet Inpainting Model**
        inpainted_img = self.inpaint_image(self.cv_img)
        print("Inpainting image type", type(inpainted_img))
        # Step 2: **Convert Image for Keypoint Prediction**
        tensor_image = F.to_tensor(inpainted_img).to(device).unsqueeze_(0)
        tensor_image = list(tensor_image)

        # Step 3: **Run Keypoint Prediction Model**
        with torch.no_grad():
            kp_model.to(device)
            kp_model.eval()
            output = kp_model(tensor_image)

        tensor_image = (tensor_image[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0].tolist() # Indexes of boxes with scores > 0.7
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], \
            output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)
        # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
        # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
        # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes
        confidence = output[0]['scores'][high_scores_idxs].detach().cpu().numpy()
        labels = output[0]['labels'][high_scores_idxs].detach().cpu().numpy()
        keypoints = []
        for idx, kps in enumerate(output[0]['keypoints'][high_scores_idxs].detach().cpu().numpy()):
            keypoints.append(list(map(int, kps[0, 0:2])) + [confidence[idx]] + [labels[idx]])

        keypoints = [torch.tensor(kp, dtype=torch.float32).to(device) if not isinstance(kp, torch.Tensor) else kp for kp in keypoints]
        keypoints = torch.stack(keypoints).to(device)
        
        unique_labels, best_keypoint_indices = torch.unique(keypoints[:, 3], return_inverse=True)
        best_scores, best_indices = torch.max(keypoints[:, 2].unsqueeze(0) * (best_keypoint_indices == torch.arange(len(unique_labels)).unsqueeze(1).cuda()), dim=1)
        keypoints = keypoints[best_indices]
        keypoints_list = keypoints.tolist()
        # print(keypoints_list)

        # keypoints_list is the list of keypoints
        keypoints_all = [[int(kp[0]), int(kp[1])] for kp in keypoints_list]

        # print(keypoints_all)


        #     # for kp in kps:
        #         # print(kp)
        #     key_points.append([list(map(int, kp[:2])) for kp in kps])
        # # print(np.array(keypoints).shape)                
        # # if len(keypoints) == 6:
        # #     keypoints.pop(2)
        # labels = []
        # for label in output[0]['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        #     labels.append(label)
        
        # unique_labels = set(labels)
        # best_indices = []
        # for ul in unique_labels:
        #     same_label_indices = [i for i, label in enumerate(labels) if label == ul]
        #     best_index = max(same_label_indices, key=lambda i: scores[high_scores_idxs[post_nms_idxs[i]]])
        #     best_indices.append(best_index)

        # keypoints_best = [keypoints_all[i] for i in best_indices]

        # if len(keypoints_all) > 2:
        #     x1, y1 = keypoints_all[1]
        #     x2, y2 = keypoints_all[2]
        #     midpoint = [(x1 + x2) // 2, (y1 + y2) // 2]
        #     # Insert the interpolated midpoint into the list at index 2
        #     keypoints_all.insert(2, midpoint)

        # print(keypoints_all)

        # print("current keypoints", keypoints_all)

        # indices = [0,1,2,3,4,5,6,7,8]
        # keypoints_all = [keypoints_all[i] for i in indices]

        # uncomment the next line for 4 feature points
        # indices = [2,3,4,5,6,8]
        # uncomment the next line 3 feature points
        indices = [3,4,6,7,8]
        # indices = [1,2,4,5,8]

        keypoints_ = [keypoints_all[i] for i in indices]

        # print(len(keypoints))
        # print(len(keypoints_all))
        # print(len(keypoints_))
        # print("exec keypoints", keypoints_)
        bboxes = []
        for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            bboxes.append(list(map(int, bbox.tolist())))

        # print("key points", keypoints)
        # img = visualize(inf_img, bboxes, key_points)
        # cv2.imwrite("/home/jc-merlab/Pictures/Data/video_results_live_b1e25_v3/out_image_" + str(i) + ".jpg", img)

        # kp = np.reshape(np.array(kp), (-1, 2))

        if len(keypoints_) == 5 and not self.executed:
            # print("first key points input", kp)
                self.corrected_x = self.first_input_estimate(self.cv_img, keypoints_)
                self.executed = True
            # print("Corrected_X", self.corrected_x)                       
            
        elif self.executed:
            # print("Executed?", self.executed)               
            self.corrected_x = self.input_estimation(self.cv_img, keypoints_, self.corrected_x) 

        print(self.corrected_x)

        kp_x = []
        kp_y = []
        for i in range(len(self.corrected_x)):
            x = np.int64(self.corrected_x[i][0])
            y = np.int64(self.corrected_x[i][1])
            kp_x.append(x)
            kp_y.append(y)

        kp = []

        if no_of_features==8:
            for i in range(len(kp_x)-1):
               kp.append(kp_x[i+1]) 
               kp.append(kp_y[i+1])
        elif no_of_features==10:
            for i in range(len(kp_x)):
                kp.append(kp_x[i]) 
                kp.append(kp_y[i])
        elif no_of_features==6:
            for i in range(len(kp_x)-2):
               kp.append(kp_x[i+2]) 
               kp.append(kp_y[i+2])
        elif no_of_features==4:
            for i in range(len(kp_x)-3):
               kp.append(kp_x[i+3]) 
               kp.append(kp_y[i+3])
        elif no_of_features==2:
            for i in range(len(kp_x)-4):
               kp.append(kp_x[i+4]) 
               kp.append(kp_y[i+4])

        # print("current keypoints", kp)

        # Uncomment the next block for 4 features
        # for i in range(len(kp_x)):
        #    kp.append(kp_x[i]) 
        #    kp.append(kp_y[i])

        kp_resp = Float64MultiArray()
        kp_resp.data = kp
        # print("keypoints", kp_resp.data)
        # print("type keypoints", type(kp_resp.data))

        # cv2.imwrite("/home/jc-merlab/Pictures/Data/video_results_full_b1e30/out_image_" + str(self.i) + ".jpg", self.cv_img)        

        self.i = self.i+1

        return dl_sim_imgResponse(self.ros_img, kp_resp)    

    def first_input_estimate(self, img, key_points):
        """ Initialize Kalman filter with first detected keypoints. """
        return self.kalman_estimate(img, key_points)

    def input_estimation(self, img, key_points, corrected_x):
        """ Correct missing keypoints using Kalman filter. """
        latest_x = np.full((5, 2), -1)  # Default to missing values

        for i, kp in enumerate(key_points):
            distances = np.linalg.norm(corrected_x - kp, axis=1)
            nearest_index = np.argmin(distances)
            latest_x[nearest_index] = kp

        return self.kalman_estimate(img, latest_x)

    def kalman_estimate(self, img, feature):
        """ Apply Kalman filtering to smooth detected keypoints. """
        updated_keypoints = []
        for i, (kf, (cx, cy)) in enumerate(zip(kalman_filters, feature)):
            measured = np.array([[np.float32(cx)], [np.float32(cy)], [0], [0]])
            predicted = kf.predict()

            updated = np.asarray(kf.update(measured))
            updated_keypoints.append([int(updated[0]), int(updated[1])])

        return np.array(updated_keypoints)

def main():
    """ Initialize ROS Node & Start Keypoint Estimation Service """
    rospy.init_node('kp_inference_with_inpainting')
    kp_obj = VideoInference()
    rospy.spin()

if __name__ == '__main__':
    main()
