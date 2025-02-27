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
from PIL import Image
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

t = torch.cuda.get_device_properties(0).total_memory
print(t)
torch.cuda.empty_cache()

r = torch.cuda.memory_reserved(0)
print(r)
a = torch.cuda.memory_allocated(0)
print(a)
# f = r-a  # free inside reserved
# **Step 1: Extract Ground Truth Keypoints (From Non-Occluded Video)**
def extract_gt_keypoints(video_path, model, device):
    cap = cv2.VideoCapture(video_path)
    gt_keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = Image.fromarray(frame)
        image = F.to_tensor(image).to(device).unsqueeze(0)

        with torch.no_grad():
            output = model(image)

        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0].tolist()
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], 
                                            output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()

        labels = output[0]['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy().tolist()
        raw_keypoints = output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()[:, 0, :2]

        sorted_keypoints = [x for _, x in sorted(zip(labels, raw_keypoints))]
          
        gt_keypoints.append(np.array(sorted_keypoints))


    cap.release()
    return gt_keypoints

def visualize_keypoints(image, predicted_keypoints, gt_keypoints, out_dir, frame_id):
    img = image.copy()  # Use the frame directly
    if torch.is_tensor(predicted_keypoints):
        predicted_keypoints = predicted_keypoints.cpu().numpy()
    if torch.is_tensor(gt_keypoints):
        gt_keypoints = gt_keypoints.cpu().numpy()

    # if np.isnan(predicted_keypoints).any():
    #         print("Handling NaN values in first frame...")
    #         # predicted_keypoints[4] = (predicted_keypoints[2] + predicted_keypoints[6]) / 2
    #         # predicted_keypoints[3] = (predicted_keypoints[2] + predicted_keypoints[4]) / 2
    #         # predicted_keypoints[5] = (predicted_keypoints[4] + predicted_keypoints[6]) / 2

    #         predicted_keypoints[7] = (predicted_keypoints[6] + predicted_keypoints[8]) / 2
            

    print("initial predicted keypoints in vis", predicted_keypoints)

    # Draw GT keypoints (always present, marked in green)
    for kp in gt_keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(img, (x, y), radius=10, color=(0, 255, 0), thickness=-1)  # Green for GT

    # Draw predicted keypoints (ignoring NaN values, marked in red)
    for kp in predicted_keypoints:
        if not np.isnan(kp).any():  # Ignore NaN keypoints
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(img, (x, y), radius=7, color=(0, 0, 255), thickness=-1)  # Red for predicted

    
    output_path = os.path.join(out_dir, f"frame_{frame_id:04d}.jpg")  # Save with frame index
    cv2.imwrite(output_path, img)


# **Step 2: UKF Implementation for Keypoints**
class KeypointUKF:
    def __init__(self, num_keypoints=9, dt=0.05):
        self.num_keypoints = num_keypoints
        self.dt = dt  
        self.state_dim = num_keypoints * 4  
        self.measurement_dim = num_keypoints * 2  

        self.points = MerweScaledSigmaPoints(n=self.state_dim, alpha=0.5, beta=2.0, kappa=0.0)

        self.ukf = UKF(dim_x=self.state_dim, dim_z=self.measurement_dim, fx=self.motion_model, 
                       hx=self.measurement_model, dt=self.dt, points=self.points)

        self.ukf.P *= 50
        self.ukf.Q = np.eye(self.state_dim) * 1.0
        self.ukf.R = np.eye(self.measurement_dim) * 0

        self.ukf.x = np.zeros(self.state_dim)
        self.initialized = False  

    # def initialize_state(self, first_keypoints):
    #     for i in range(self.num_keypoints):
    #         idx = i * 4
    #         self.ukf.x[idx] = first_keypoints[i, 0]  
    #         self.ukf.x[idx + 1] = first_keypoints[i, 1]  
    #         self.ukf.x[idx + 2] = 0  
    #         self.ukf.x[idx + 3] = 0  

    #     self.initialized = True

    def initialize_state(self, first_keypoints):
        """ Initializes UKF state from the first keypoint prediction (9 keypoints, each with x, y, vx, vy). """
        first_keypoints = np.array(first_keypoints)  # Convert to NumPy for manipulation

        # Check for NaNs in the first frame
        # if np.isnan(first_keypoints).any():
        #     print("Handling NaN values in first frame...")
        #     first_keypoints[4] = (first_keypoints[2] + first_keypoints[6]) / 2
        #     first_keypoints[3] = (first_keypoints[2] + first_keypoints[4]) / 2
        #     first_keypoints[5] = (first_keypoints[4] + first_keypoints[6]) / 2

        if np.isnan(first_keypoints).any:
            first_keypoints[7] = (first_keypoints[6] + first_keypoints[8]) / 2

        # Initialize UKF with corrected values
        for i in range(self.num_keypoints):
            idx = i * 4
            self.ukf.x[idx] = first_keypoints[i, 0]  # x
            self.ukf.x[idx + 1] = first_keypoints[i, 1]  # y
            self.ukf.x[idx + 2] = 0  # Initial velocity x
            self.ukf.x[idx + 3] = 0  # Initial velocity y

        self.initialized = True

    def motion_model(self, x, dt):
        new_x = np.copy(x)
        for i in range(self.num_keypoints):
            idx = i * 4
            new_x[idx] += x[idx + 2] * dt  
            new_x[idx + 1] += x[idx + 3] * dt  
        return new_x

    def measurement_model(self, x):
        z = np.zeros(self.measurement_dim)
        for i in range(self.num_keypoints):
            z[i * 2] = x[i * 4]  
            z[i * 2 + 1] = x[i * 4 + 1]  
        return z

    def update(self, observed_keypoints, labels, mask):
        observed_keypoints = observed_keypoints.cpu().numpy()
        full_measurement = np.full((self.num_keypoints, 2), np.nan)

        for i, lbl in enumerate(labels):
            full_measurement[lbl - 1] = observed_keypoints[i]  

        z_flat = full_measurement.flatten()

        if not self.initialized:
            self.initialize_state(full_measurement)
            return full_measurement  

        self.ukf.predict()

        predicted_keypoints = self.measurement_model(self.ukf.x).reshape(self.num_keypoints, 2)
        for i in range(self.num_keypoints):
            if mask[i] == 0:  
                z_flat[i * 2] = predicted_keypoints[i, 0]  
                z_flat[i * 2 + 1] = predicted_keypoints[i, 1]  

        self.ukf.update(z_flat)

        filtered_x = self.ukf.x[::4]
        filtered_y = self.ukf.x[1::4]
        return np.stack((filtered_x, filtered_y), axis=1)


# **Step 3: Extract Predicted Keypoints (From Occluded Video)**
def extract_predicted_keypoints(video_path, model, keypoint_filter, device, output_path, gt_keypoints):
    cap = cv2.VideoCapture(video_path)
    predicted_keypoints = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = Image.fromarray(frame)
        image = F.to_tensor(image).to(device).unsqueeze(0)

        with torch.no_grad():
            output = model(image)

        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0].tolist()
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], 
                                            output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()

        labels = output[0]['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy().tolist()
        raw_keypoints = output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()[:, 0, :2]
        raw_keypoints = torch.tensor(raw_keypoints, dtype=torch.float32)

        all_labels = set(range(1, 10))  
        detected_labels = set(labels)
        missing_labels = all_labels - detected_labels  

        mask = np.zeros(9, dtype=np.int8)
        for lbl in detected_labels:
            mask[lbl - 1] = 1  

        filtered_keypoints = keypoint_filter.update(raw_keypoints, labels, mask)
        filtered_keypoints = np.array(filtered_keypoints)

        # Visualize both GT and predicted keypoints in the same frame
        visualize_keypoints(frame, filtered_keypoints, gt_keypoints[frame_id], output_path, frame_id)

        predicted_keypoints.append(filtered_keypoints)
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    return predicted_keypoints


# **Step 4: Compute Error Metrics**
def compute_errors(gt_keypoints, pred_keypoints):
    errors_per_frame = []
    errors_per_keypoint = {i: [] for i in range(9)}

    for frame_idx in range(len(gt_keypoints)):
        gt = gt_keypoints[frame_idx]
        pred = pred_keypoints[frame_idx]

        frame_errors = np.linalg.norm(gt - pred, axis=1)
        errors_per_frame.append(np.mean(frame_errors))

        for i in range(9):
            errors_per_keypoint[i].append(frame_errors[i])

    return errors_per_frame, errors_per_keypoint


# **Step 5: Visualization**
def visualize_errors(errors_per_frame, errors_per_keypoint):
    frames = list(range(len(errors_per_frame)))

    plt.figure(figsize=(16,12))
    plt.plot(frames, errors_per_frame, label="Mean Error per Frame", color='b')
    plt.xlabel("Frame", fontsize=28)
    plt.ylabel("Mean Error (pixels)", fontsize=28)
    plt.title("Frame-wise Keypoint Prediction Error", fontsize=28)
    plt.xticks(fontsize=18)  # Increase font size of x-axis tick values
    plt.yticks(fontsize=18)  # Increase font size of y-axis tick values
    plt.legend(fontsize=20)
    plt.grid()
    plt.show()

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle("Per-Keypoint Error Histograms", fontsize=32)

    for i, ax in enumerate(axes.flatten()):
        ax.hist(errors_per_keypoint[i], bins=20, color='g', alpha=0.7)
        ax.set_title(f"Keypoint {i+1} Error")

    plt.tight_layout()
    plt.show()

def visualize_individual_keypoint_errors(errors_per_keypoint):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Per-Keypoint Error Over Frames", fontsize=20)

    frames = list(range(len(errors_per_keypoint[0])))

    # Compute global min and max values for consistent y-axis limits
    min_error = min(min(errors) for errors in errors_per_keypoint.values())
    max_error = max(max(errors) for errors in errors_per_keypoint.values())

    for i, ax in enumerate(axes.flatten()):
        ax.plot(frames, errors_per_keypoint[i], label=f"Keypoint {i+1}", color='r')
        ax.set_title(f"Keypoint {i+1}", fontsize=14)
        ax.set_xlabel("Frame", fontsize=12)
        ax.set_ylabel("Error (pixels)", fontsize=12)
        ax.legend()
        ax.grid()

        print(max_error)
        # Apply the same y-axis limit for consistency
        ax.set_ylim(0, 150)

    plt.tight_layout()
    plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

weights_path = '/home/jc-merlab/Pictures/Data/trained_models/keypointsrcnn_planning_b1_e50_v8.pth'
model = torch.load(weights_path).to(device)
# model = get_model(num_keypoints=6, weights_path=weights_path)
# model.load_state_dict(torch.load('keypointsrcnn_weights.pth'))

model.to(device)
model.eval()

# **Final Execution**
non_occluded_video_path = "/home/jc-merlab/Pictures/Test_Data/vid_occ_gt/exp_03/output_video.avi"
occluded_video_path = "/home/jc-merlab/Pictures/Test_Data/vid_occ_pred/exp_03/output_video.avi"
output_path = '/home/jc-merlab/Pictures/Test_Data/vid_occ_kprcnn_gt_ukf_03/'

gt_keypoints = extract_gt_keypoints(non_occluded_video_path, model, device)
keypoint_filter = KeypointUKF()
pred_keypoints = extract_predicted_keypoints(occluded_video_path, model, keypoint_filter, device, output_path, gt_keypoints)

errors_per_frame, errors_per_keypoint = compute_errors(gt_keypoints, pred_keypoints)
visualize_errors(errors_per_frame, errors_per_keypoint)
visualize_individual_keypoint_errors(errors_per_keypoint)
