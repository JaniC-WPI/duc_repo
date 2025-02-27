import os
from os import listdir
import pandas as pd
import numpy as np
import glob
import cv2
import json
from os.path import expanduser
import splitfolders
import shutil
from define_path import Def_Path

from tqdm import tqdm

import torch 
import torchvision
from torchvision import models
from torchvision.models.detection.rpn import AnchorGenerator
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchsummary import summary

from sklearn.model_selection import train_test_split

import albumentations as A # Library for augmentations

import matplotlib.pyplot as plt 
from PIL import Image

import transforms, utils, engine, train
from utils import collate_fn
from engine import train_one_epoch, evaluate
import time


t = torch.cuda.get_device_properties(0).total_memory
print(t)
torch.cuda.empty_cache()

r = torch.cuda.memory_reserved(0)
print(r)
a = torch.cuda.memory_allocated(0)
print(a)
# f = r-a  # free inside reserved
import cv2
import numpy as np

class KalmanFilter2D:
    def __init__(self, initial_measurement):
        """
        Simple 2D Kalman Filter for keypoint tracking.
        State vector: [x, y, vx, vy]^T
        Measurement: [x, y]^T
        """
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0  # time step
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0],
                                               [0, 1, 0, dt],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 0.1
        # Initialize state with measurement and zero velocity
        self.kf.statePost = np.array([[initial_measurement[0]],
                                      [initial_measurement[1]],
                                      [0],
                                      [0]], np.float32)
    
    def update(self, measurement):
        """
        Predict the next state; update with measurement if available.
        measurement: (x, y) or None.
        Returns predicted (or updated) [x, y].
        """
        prediction = self.kf.predict()  # shape (4,1)
        pred_pos = prediction[:2].flatten()
        if measurement is not None:
            meas = np.array([[measurement[0]], [measurement[1]]], np.float32)
            self.kf.correct(meas)
            return meas.flatten()
        else:
            return pred_pos

# Global variable for Kalman filters (one per keypoint)
global_kf_filters = None

def apply_kalman_filters(denorm_keypoints, kf_filters=None, missing_thresh=1e-3):
    """
    denorm_keypoints: numpy array of shape (9,2)
    kf_filters: a list of KalmanFilter2D objects (if None, will initialize)
    missing_thresh: if the norm of a keypoint is below this threshold, treat as missing.
    
    Returns:
       smoothed: numpy array of shape (9,2)
       kf_filters: updated list of filters.
    """
    num = denorm_keypoints.shape[0]
    if kf_filters is None:
        kf_filters = [None] * num
    smoothed = np.zeros_like(denorm_keypoints)
    for i in range(num):
        meas = denorm_keypoints[i]  # (x, y)
        if np.linalg.norm(meas) < missing_thresh:
            meas_in = None
        else:
            meas_in = meas
        if kf_filters[i] is None and meas_in is not None:
            kf_filters[i] = KalmanFilter2D(meas_in)
            smoothed[i] = meas_in
        elif kf_filters[i] is not None:
            smoothed[i] = kf_filters[i].update(meas_in)
        else:
            smoothed[i] = np.array([0, 0], dtype=np.float32)
    return smoothed, kf_filters

# --------------------------
# New Visualization Function
# --------------------------
global_kf_filters = None

def process_frame_with_kalman(frame, denorm_keypoints):
    global global_kf_filters
    # Draw original (raw) denormalized keypoints in blue for comparison.
    frame_orig = frame.copy()
    for (x, y) in denorm_keypoints:
        cv2.circle(frame_orig, (int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)  # Blue circles (BGR: Blue= (255,0,0))
    
    # Apply Kalman filtering to obtain smoothed keypoints.
    smoothed_keypoints, global_kf_filters = apply_kalman_filters(denorm_keypoints, kf_filters=global_kf_filters)
    
    # Draw smoothed keypoints in red.
    frame_combined = frame.copy()
    for (x, y) in denorm_keypoints:
        cv2.circle(frame_combined, (int(x), int(y)), radius=8, color=(255, 0, 0), thickness=-1)  # Original in blue
    for (x, y) in smoothed_keypoints:
        cv2.circle(frame_combined, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)  # Smoothed in red
    
    return smoothed_keypoints, frame_combined

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

weights_path = '/home/jc-merlab/Pictures/Data/trained_models/keypointsrcnn_planning_b1_e50_v8.pth'
model = torch.load(weights_path).to(device)
# model = get_model(num_keypoints=6, weights_path=weights_path)
# model.load_state_dict(torch.load('keypointsrcnn_weights.pth'))

model.to(device)
model.eval()

# Assuming the necessary imports are done
# Assuming the model is loaded and device is set as in your initial code

# Specify input and output folders
input_folder = '/home/jc-merlab/Pictures/Data/occ_panda_phys_test_data/'
output_path = '/home/jc-merlab/Pictures/Test_Data/vid_occ_kp/'

# Check if output folders exist, create them if not
os.makedirs(output_path, exist_ok=True)

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np
import torch

class KeypointUKF:
    def __init__(self, num_keypoints=9, dt=1.0):
        self.num_keypoints = num_keypoints
        self.dt = dt  # Time step between frames

        # UKF State Size: Each keypoint has (x, y, vx, vy), so total 36D
        self.state_dim = self.num_keypoints * 4
        self.measurement_dim = self.num_keypoints * 2  # Only x, y

        # Sigma Points for UKF
        self.points = MerweScaledSigmaPoints(n=self.state_dim, alpha=0.5, beta=2.0, kappa=0.0)

        # UKF Initialization
        self.ukf = UKF(dim_x=self.state_dim, dim_z=self.measurement_dim, fx=self.motion_model, 
                       hx=self.measurement_model, dt=self.dt, points=self.points)

        # Process and Measurement Noise
        self.ukf.P *= 10 # Initial uncertainty
        self.ukf.Q = np.eye(self.state_dim) * 0.2  # Process noise
        self.ukf.R = np.eye(self.measurement_dim) * 15  # Measurement noise

        # Initialize state vector (positions and velocities)
        self.ukf.x = np.zeros(self.state_dim)

        self.initialized = False  # UKF should not filter first frame

    def initialize_state(self, first_keypoints):
        """ Initializes UKF state from the first keypoint prediction (9 keypoints, each with x, y, vx, vy). """
        # first_keypoints = first_keypoints.cpu().numpy()

        for i in range(self.num_keypoints):
            idx = i * 4
            self.ukf.x[idx] = first_keypoints[i, 0]  # x
            self.ukf.x[idx + 1] = first_keypoints[i, 1]  # y
            self.ukf.x[idx + 2] = 0  # Initial velocity x
            self.ukf.x[idx + 3] = 0  # Initial velocity y

        self.initialized = True

    def motion_model(self, x, dt):
        """ State transition function assuming constant velocity model """
        new_x = np.copy(x)

        for i in range(self.num_keypoints):
            idx = i * 4
            new_x[idx] += x[idx + 2] * dt  # x = x + vx * dt
            new_x[idx + 1] += x[idx + 3] * dt  # y = y + vy * dt
            # Velocity remains unchanged

        return new_x

    def measurement_model(self, x):
        """ Extracts x, y positions from the full state vector """
        z = np.zeros(self.measurement_dim)

        for i in range(self.num_keypoints):
            z[i * 2] = x[i * 4]  # x
            z[i * 2 + 1] = x[i * 4 + 1]  # y

        return z

    def update(self, observed_keypoints, labels, mask):
        """ Updates UKF with detected keypoints and predicts missing ones. """
        observed_keypoints = observed_keypoints.cpu().numpy()

        # Convert to full measurement array (fill missing with NaN)
        full_measurement = np.full((self.num_keypoints, 2), np.nan)

        for i, lbl in enumerate(labels):
            full_measurement[lbl - 1] = observed_keypoints[i]  # Labels are 1-indexed

        # Flatten measurement vector for UKF update
        z_flat = full_measurement.flatten()

        # **First Frame Handling**
        if not self.initialized:
            self.initialize_state(full_measurement)
            return full_measurement  # Directly return first frame output

        # Predict the next state
        self.ukf.predict()

        # Fill missing keypoints using UKF prediction
        predicted_keypoints = self.measurement_model(self.ukf.x).reshape(self.num_keypoints, 2)
        for i in range(self.num_keypoints):
            if mask[i] == 0:  # If keypoint is missing
                z_flat[i * 2] = predicted_keypoints[i, 0]  # x
                z_flat[i * 2 + 1] = predicted_keypoints[i, 1]  # y

        # Update UKF with corrected observations
        self.ukf.update(z_flat)

        # Extract filtered keypoints (x, y) from UKF
        filtered_x = self.ukf.x[::4]
        filtered_y = self.ukf.x[1::4]
        filtered_keypoints = np.stack((filtered_x, filtered_y), axis=1)

        return filtered_keypoints
    
def visualize_keypoints(image, predicted_keypoints, out_dir, frame_id):
    img = image.copy()  # Use the frame directly
    if torch.is_tensor(predicted_keypoints):
        predicted_keypoints = predicted_keypoints.cpu().numpy()

    for kp in predicted_keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(img, (x, y), radius=8, color=(255, 0, 0), thickness=-1)
    
    output_path = os.path.join(out_dir, f"frame_{frame_id:04d}.jpg")  # Save with frame index
    cv2.imwrite(output_path, img)



# print(type(model))
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

cap = cv2.VideoCapture('/home/jc-merlab/Pictures/Test_Data/vid_occ_pred/exp_01/output_video.avi')
if not cap.isOpened():
    print("Error opening video stream or file")

frame_id = 0
keypoint_filter = KeypointUKF()  # Initialize UKF
while(cap.isOpened()):
  # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:        
#         img = cv2.imread(frame)
        image = Image.fromarray(frame)

        image = F.to_tensor(image).to(device)
        image.unsqueeze_(0)
        image = list(image)
    
        with torch.no_grad():
            model.to(device)
            model.eval()
            start = time.time()
            output = model(image)
            stop = time.time()
            print("time", (stop - start))

        image = (image[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
        image_vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        scores = output[0]['scores'].detach().cpu().numpy()

        high_scores_idxs = np.where(scores > 0.7)[0].tolist() # Indexes of boxes with scores > 0.7
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

        # Below, in output[0]['keypoints'][high_scores_idxs][post_nms_idxs] and output[0]['boxes'][high_scores_idxs][post_nms_idxs]
        # Firstly, we choose only those objects, which have score above predefined threshold. This is done with choosing elements with [high_scores_idxs] indexes
        # Secondly, we choose only those objects, which are left after NMS is applied. This is done with choosing elements with [post_nms_idxs] indexes
        # labels = []
        # for label in output[0]['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        #     labels.append(label)

        # print(labels)
        # raw_keypoints_list = []
        # for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
        #     raw_keypoints_list.append(list(map(int, kps[0,0:2])))

        # raw_keypoints = [x for _,x in sorted(zip(labels,raw_keypoints_list))]

        # raw_keypoints = torch.tensor(raw_keypoints, dtype=torch.float32)

        # print(raw_keypoints.shape)

        labels = output[0]['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy().tolist()
        raw_keypoints = output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()[:, 0, :2]
        raw_keypoints = torch.tensor(raw_keypoints, dtype=torch.float32)

        # **Step 1: Identify Missing Keypoints**
        all_labels = set(range(1, 10))  # {1,2,3,...,9}
        detected_labels = set(labels)
        missing_labels = all_labels - detected_labels  # Which keypoints are missing?

        # **Step 2: Create a Mask (1 = Present, 0 = Missing)**
        mask = np.zeros(9, dtype=np.int8)
        for lbl in detected_labels:
            mask[lbl - 1] = 1  # Mark detected keypoints

        bboxes = []
        for bbox in output[0]['boxes'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            bboxes.append(list(map(int, bbox.tolist())))

        # **Step 3: Apply UKF**
        filtered_keypoints = keypoint_filter.update(raw_keypoints, labels, mask)

        # **Step 4: Visualize and Save the Output**
        visualize_keypoints(frame, filtered_keypoints, output_path, frame_id)

        # kf_keypoints, global_kf_filters = apply_kalman_filters(raw_keypoints, kf_filters=global_kf_filters)
        
        # Save the visualization frame
        # cv2.imwrite(os.path.join(output_path, f"frame_{i:04d}.jpg"), frame_with_kf)

            
    else:
        break
        
    frame_id += 1
    
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()