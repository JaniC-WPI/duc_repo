#!/usr/bin/env python
# coding: utf-8

import os
import time
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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
from datetime import datetime

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
from torch.cuda.amp import GradScaler, autocast

import albumentations as A  # Library for augmentations

import matplotlib.pyplot as plt 
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from torch.autograd import Variable
import torch_geometric.nn as pyg
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch_geometric.data import Data

# Print CUDA info (for debugging GPU memory)
t = torch.cuda.get_device_properties(0).total_memory
print(t)
torch.cuda.empty_cache()
r = torch.cuda.memory_reserved(0)
print(r)
a = torch.cuda.memory_allocated(0)
print(a)

weights_path = '/home/jc-merlab/Pictures/Data/trained_models/keypointsrcnn_planning_b1_e50_v8.pth'
n_nodes = 9

# Generalize home directory.
path = Def_Path()
parent_path = "/home/schatterjee/lama/kprcnn_panda/"
root_dir = parent_path + "occ_panda_physical_dataset/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def train_transform():
    return A.Compose([
        A.Sequential([
            A.RandomRotate90(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1),
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'),
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels'])
    )

def train_test_split(src_dir):
    dst_dir_img = src_dir + "images"
    dst_dir_anno = src_dir + "annotations"
    if not (os.path.exists(dst_dir_img) and os.path.exists(dst_dir_anno)):
        os.mkdir(dst_dir_img)
        os.mkdir(dst_dir_anno)
    for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
        shutil.copy(jpgfile, dst_dir_img)
    for jsonfile in glob.iglob(os.path.join(src_dir, "*.json")):
        shutil.copy(jsonfile, dst_dir_anno)
    output = parent_path + "split_folder_output_occ_sage" + "-" + path.year + "-" + path.month + "-" + path.day
    splitfolders.ratio(src_dir, output=output, seed=42, ratio=(0.95, 0.025, 0.025), group_prefix=None, move=False)
    shutil.rmtree(dst_dir_img)
    shutil.rmtree(dst_dir_anno)
    return output  

class KPDataset(Dataset):
    def __init__(self, root, transform=None, demo=False):                
        self.root = root
        self.transform = transform
        self.demo = demo
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))
    
    def __getitem__(self, idx):
        img_file = self.imgs_files[idx]
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        annotations_path = os.path.join(self.root, "annotations", self.annotations_files[idx])
        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        with open(annotations_path) as f:
            data = json.load(f)
            bboxes_original = data['bboxes']
            keypoints_original = data['keypoints']
            bboxes_labels_original = ['base_joint','joint2','joint3','joint4',
                                      'joint5','joint6','joint7','joint8','joint9']
        if self.transform:   
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]
            transformed = self.transform(image=img_original, bboxes=bboxes_original, 
                                         bboxes_labels=bboxes_labels_original, keypoints=keypoints_original_flattened)
            img = transformed['image']
            bboxes = transformed['bboxes']
            keypoints_transformed_unflattened = np.reshape(np.array(transformed['keypoints']), (-1,1,2)).tolist()
            keypoints = []
            for o_idx, obj in enumerate(keypoints_transformed_unflattened):
                obj_keypoints = []
                for k_idx, kp in enumerate(obj):
                    obj_keypoints.append(kp + [keypoints_original[o_idx][k_idx][2]])
                keypoints.append(obj_keypoints)
        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)       
        target = {}
        labels = [1,2,3,4,5,6,7,8,9]
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bboxes[:,3]-bboxes[:,1])*(bboxes[:,2]-bboxes[:,0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)
        img = F.to_tensor(img)        
        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target_original["image_id"] = torch.tensor([idx])
        target_original["area"] = (bboxes_original[:,3]-bboxes_original[:,1])*(bboxes_original[:,2]-bboxes_original[:,0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor(keypoints_original, dtype=torch.float32)
        img_original = F.to_tensor(img_original)
        if self.demo:
            return img, target, img_original, target_original, img_file
        else:
            return img, target, img_file
    
    def __len__(self):
        return len(self.imgs_files)

class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = x.cuda()
        edge_index = edge_index.cuda()
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.fc(x)
        return x

def calculate_distance_angle(kp1, kp2):
    dx = kp2[0] - kp1[0]
    dy = kp2[1] - kp1[1]
    dx, dy = torch.tensor(dx).to(device), torch.tensor(dy).to(device)
    distance = torch.sqrt(dx ** 2 + dy ** 2)
    angle = torch.atan2(dy, dx)
    return distance, angle

def calculate_gt_distances_angles(keypoints_gt):
    batch_size, num_keypoints, num_dims = keypoints_gt.shape
    assert num_keypoints == n_nodes and num_dims == 2, "keypoints_gt must have shape (batch_size, 9, 2)"
    distances_angles = []
    for b in range(batch_size):
        batch_distances_angles = torch.zeros((num_keypoints, 4), dtype=torch.float32).to(device)
        for i in range(num_keypoints):
            current_kp = keypoints_gt[b, i]
            next_i = (i + 1) % num_keypoints
            prev_i = (i - 1 + num_keypoints) % num_keypoints
            dist, angle = calculate_distance_angle(current_kp, keypoints_gt[b, next_i])
            batch_distances_angles[i, 0] = dist
            batch_distances_angles[i, 1] = angle
            dist, angle = calculate_distance_angle(current_kp, keypoints_gt[b, prev_i])
            batch_distances_angles[i, 2] = dist
            batch_distances_angles[i, 3] = angle
        distances_angles.append(batch_distances_angles)
    distances_angles = torch.stack(distances_angles)
    return distances_angles

# --------------------------
# Kalman Filter Implementation
# --------------------------
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
# KeypointPipeline for Inference
# --------------------------
class KeypointPipeline(nn.Module):
    def __init__(self, weights_path):
        super(KeypointPipeline, self).__init__()  
        self.keypoint_model = torch.load(weights_path).to(device)
        self.graph_gcn = GraphGCN(8, 1024, 4)
        
    def process_model_output(self, output):
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.01)[0].tolist()
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs],
                                              output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()
        confidence = output[0]['scores'][high_scores_idxs].detach().cpu().numpy()
        labels = output[0]['labels'][high_scores_idxs].detach().cpu().numpy()
        keypoints = []
        for idx, kps in enumerate(output[0]['keypoints'][high_scores_idxs].detach().cpu().numpy()):
            keypoints.append(list(map(int, kps[0, 0:2])) + [confidence[idx]] + [labels[idx]])
        keypoints = [torch.tensor(kp, dtype=torch.float32).to(device) for kp in keypoints]
        keypoints = torch.stack(keypoints).to(device)
        unique_labels, best_keypoint_indices = torch.unique(keypoints[:, 3], return_inverse=True)
        best_scores, best_indices = torch.max(keypoints[:, 2].unsqueeze(0) * 
                                               (best_keypoint_indices == torch.arange(len(unique_labels)).unsqueeze(1).cuda()),
                                               dim=1)
        keypoints = keypoints[best_indices]
        return keypoints
    
    def fill_missing_keypoints(self, keypoints, image_width, image_height):
        keypoints_dict = {int(kp[3]): kp for kp in keypoints}
        complete_keypoints = []
        labels = [int(kp[3]) for kp in keypoints]
        all_labels = set(range(1, 10))
        missing_labels = list(all_labels - set(labels))
        missing_labels.sort()
        for i in range(1, 10):
            if i in keypoints_dict:
                complete_keypoints.append(keypoints_dict[i].tolist())
            else:
                prev_label = i - 1 if i > 1 else 9
                next_label = i + 1 if i < 9 else 1
                prev_kp = keypoints_dict.get(prev_label, [image_width/2, image_height/2, 0, prev_label])
                next_kp = keypoints_dict.get(next_label, [image_width/2, image_height/2, 0, next_label])
                if next_label in missing_labels:
                    next_kp = [image_width/2, image_height/2, 0, next_label]
                avg_x = (prev_kp[0] + next_kp[0]) / 2
                avg_y = (prev_kp[1] + next_kp[1]) / 2
                complete_keypoints.append([avg_x, avg_y, 0, i])
        return torch.tensor(complete_keypoints, dtype=torch.float32).to(device)
    
    def normalize_keypoints(self, keypoints, image_width, image_height):
        keypoints[:,0] = (keypoints[:,0] - image_width/2) / (image_width/2)
        keypoints[:,1] = (keypoints[:,1] - image_height/2) / (image_height/2)
        return keypoints
    
    def keypoints_to_graph(self, keypoints, image_width, image_height):
        node_features = []
        for i, kp in enumerate(keypoints):
            x, y, conf, label = kp
            prev_kp = keypoints[i-1]
            next_kp = keypoints[(i+1) % len(keypoints)]
            dist_next, angle_next = calculate_distance_angle([x,y], next_kp[:2])
            dist_prev, angle_prev = calculate_distance_angle([x,y], prev_kp[:2])
            node_features.append([x, y, conf, label, dist_next, angle_next, dist_prev, angle_prev])
        node_features = torch.tensor(node_features, dtype=torch.float32).to(device)
        edge_index = torch.tensor([[i, (i+1)%len(keypoints)] for i in range(len(keypoints))] +
                                    [[(i+1)%len(keypoints), i] for i in range(len(keypoints))],
                                    dtype=torch.long).t().contiguous().to(device)
        return Data(x=node_features, edge_index=edge_index)
    
    def forward(self, imgs):
        keypoint_model_training = self.keypoint_model.training
        self.keypoint_model.eval()
        with torch.no_grad():
            batch_outputs = [self.keypoint_model(img.unsqueeze(0).to(device)) for img in imgs]
        self.keypoint_model.train(mode=keypoint_model_training)
        batch_labeled_keypoints = [self.process_model_output(output) for output in batch_outputs]
        for idx, keypoints in enumerate(batch_labeled_keypoints):
            image_width, image_height = 640, 480
            filled_keypoints = self.fill_missing_keypoints(keypoints, image_width, image_height)
            normalized_keypoints = self.normalize_keypoints(filled_keypoints, image_width, image_height)
            batch_labeled_keypoints[idx] = normalized_keypoints
        all_graphs = [self.keypoints_to_graph(keypoints, 640, 480) for keypoints in batch_labeled_keypoints]
        all_predictions = [self.graph_gcn(graph.x, graph.edge_index) for graph in all_graphs]
        final_predictions = torch.stack(all_predictions)
        return final_predictions

def kgnn2d_loss(gt_keypoints, pred_keypoints, gt_distances_next, gt_angles_next, 
                 gt_distances_prev, gt_angles_prev, pred_distances_next, pred_angles_next, 
                 pred_distances_prev, pred_angles_prev):
    keypoints_loss = func.mse_loss(pred_keypoints, gt_keypoints)
    prev_distances_loss = func.mse_loss(pred_distances_prev, gt_distances_prev)
    prev_angles_loss = func.mse_loss(pred_angles_prev, gt_angles_prev)
    next_distances_loss = func.mse_loss(pred_distances_next, gt_distances_next)
    next_angles_loss = func.mse_loss(pred_angles_next, gt_angles_next)
    return keypoints_loss + prev_distances_loss + prev_angles_loss + next_distances_loss + next_angles_loss

def process_batch_keypoints(target_dicts):
    batch_size = len(target_dicts)
    keypoints_list = []
    for dict_ in target_dicts:
        keypoints = dict_['keypoints'].squeeze(1).to(device)
        xy_coords = keypoints[:, :2]
        keypoints_list.append(xy_coords)
    keypoints_gt = torch.stack(keypoints_list).float().cuda()
    return keypoints_gt

def reorder_batch_keypoints(batch_keypoints):
    batch_size, num_keypoints, num_features = batch_keypoints.shape
    reordered_keypoints_batch = []
    for i in range(batch_size):
        normalized_keypoints = batch_keypoints[i]
        reordered_normalized_keypoints = torch.zeros(num_keypoints, 2, device=batch_keypoints.device)
        rounded_labels = torch.round(normalized_keypoints[:, -1]).int()
        used_indices = []
        for label in range(1, 10):
            valid_idx = (rounded_labels == label).nonzero(as_tuple=True)[0]
            if valid_idx.numel() > 0:
                reordered_normalized_keypoints[label - 1] = normalized_keypoints[valid_idx[0], :2]
            else:
                invalid_idx = ((rounded_labels < 1) | (rounded_labels > 9)).nonzero(as_tuple=True)[0]
                invalid_idx = [idx for idx in invalid_idx if idx not in used_indices]
                if invalid_idx:
                    reordered_normalized_keypoints[label - 1] = normalized_keypoints[invalid_idx[0], :2]
                    used_indices.append(invalid_idx[0])
        reordered_keypoints_batch.append(reordered_normalized_keypoints)
    return torch.stack(reordered_keypoints_batch)

def denormalize_keypoints(batch_keypoints, width=640, height=480):
    denormalized_keypoints = []
    for kp in batch_keypoints:
        denormalized_x = (kp[:, 0] * (width/2)) + (width/2)
        denormalized_y = (kp[:, 1] * (height/2)) + (height/2)
        denormalized_kp = torch.stack((denormalized_x, denormalized_y), dim=1)
        denormalized_keypoints.append(denormalized_kp)
    denormalized_keypoints = torch.stack(denormalized_keypoints)
    return denormalized_keypoints

import cv2
import os
import torch
from torchvision.transforms import functional as F
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instantiate the model
model = KeypointPipeline(weights_path)
model = model.to(device)

# Load the checkpoint
checkpoint_path = '/home/jc-merlab/Pictures/Data/trained_models/gcn_ckpt_v2/kprcnn_occ_gcn_ckpt_b128e17.pth'
checkpoint = torch.load(checkpoint_path)
model_state_dict = checkpoint['model_state_dict']
model.load_state_dict(model_state_dict)
model.eval()

def prepare_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img).to(device)
    return img_tensor

def load_ground_truth(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    ground_truth_keypoints = [[int(kp[0][0]), int(kp[0][1]), kp[0][2]] for kp in data['keypoints']]
    return ground_truth_keypoints

def draw_lines_between_keypoints(image, keypoints, color=(255,255,255)):
    for i in range(len(keypoints)-1):
        start_point = (int(keypoints[i][0]), int(keypoints[i][1]))
        end_point = (int(keypoints[i+1][0]), int(keypoints[i+1][1]))
        cv2.line(image, start_point, end_point, color, thickness=2)
    for x,y in keypoints:
        cv2.circle(image, (x,y), radius=5, color=(0,0,255), thickness=-1)

def predict(model, img_tensor):
    with torch.no_grad():
        KGNN2D = model([img_tensor])
    print("Predicted keypoints", KGNN2D)
    return KGNN2D

def postprocess_keypoints(keypoints, width=640, height=480):
    denormalized_keypoints = denormalize_keypoints(keypoints, width, height)
    return denormalized_keypoints

def visualize_keypoints_with_ground_truth(image_path, predicted_keypoints, ground_truth_keypoints, out_dir):
    img = cv2.imread(image_path)
    if torch.is_tensor(predicted_keypoints):
        predicted_keypoints = predicted_keypoints.cpu().numpy()
    for kp in predicted_keypoints[0]:
        x,y = int(kp[0]), int(kp[1])
        cv2.circle(img, (x,y), radius=8, color=(255,0,0), thickness=-1)
    for x,y,_ in ground_truth_keypoints:
        cv2.circle(img, (x,y), radius=6, color=(0,0,255), thickness=-1)
    filename = os.path.basename(image_path)
    output_path = os.path.join(out_dir, filename)
    cv2.imwrite(output_path, img)

def visualize_keypoints(image, predicted_keypoints, out_dir, frame_id):
    img = image.copy()  # Use the frame directly
    if torch.is_tensor(predicted_keypoints):
        predicted_keypoints = predicted_keypoints.cpu().numpy()
    for kp in predicted_keypoints[0]:
        x,y = int(kp[0]), int(kp[1])
        cv2.circle(img, (x,y), radius=8, color=(255,0,0), thickness=-1)
    output_path = os.path.join(out_dir, f"frame_{frame_id:04d}.jpg")
    cv2.imwrite(output_path, img)

def calculate_accuracy(predicted_keypoints, ground_truth_keypoints, margin=10):
    correct = 0
    total = len(ground_truth_keypoints)
    correct_invisible = 0
    total_invisible = 0
    for pred_kp, gt_kp in zip(predicted_keypoints[0], ground_truth_keypoints):
        pred_kp = pred_kp.cpu().numpy()
        dist = np.linalg.norm(np.array(pred_kp[:2]) - np.array(gt_kp[:2]))
        if gt_kp[2] == 0:
            total_invisible += 1
            if dist <= margin:
                correct_invisible += 1
        else:
            if dist <= margin:
                correct += 1
    correct_total = correct + correct_invisible
    accuracy = (correct_total / total) * 100
    invisible_accuracy = (correct_invisible / total_invisible)*100 if total_invisible > 0 else 0
    return accuracy, invisible_accuracy, total_invisible

# --------------------------
# Inference with Kalman Filter Integration
# --------------------------
# global_kf_filters will store the list of 9 KalmanFilter2D objects across frames.
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

def process_video(folder_path, output_path):
    cap = cv2.VideoCapture(folder_path)
    frame_id = 0
    global global_kf_filters
    global_kf_filters = None  # Reset at the start
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(frame)
        img_tensor = F.to_tensor(image).to(device)
        KGNN2D = predict(model, img_tensor)
        ordered_keypoints = reorder_batch_keypoints(KGNN2D)
        denormalized_keypoints = postprocess_keypoints(ordered_keypoints)
        # denormalized_keypoints is of shape (1,9,2); take the first batch element:
        denorm = denormalized_keypoints[0].cpu().numpy()  # shape (9,2)
        smoothed_keypoints, frame_with_kf = process_frame_with_kalman(frame, denorm)
        cv2.imwrite(os.path.join(output_path, f"frame_{frame_id:04d}.jpg"), frame_with_kf)
        frame_id += 1
    cap.release()
    cv2.destroyAllWindows()

def process_folder(folder_path, output_path, output_path_line):
    accuracies = []
    invisible_accuracies = []
    total_invisible_keypoints = 0
    total_inference_time = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img_tensor = prepare_image(image_path).to(device)
            start_time = time.time()
            KGNN2D = predict(model, img_tensor)
            end_time = time.time()
            inference_time = end_time - start_time
            print(f"Inference time for {filename}: {inference_time:.4f} seconds")
            ordered_keypoints = reorder_batch_keypoints(KGNN2D)
            denormalized_keypoints = postprocess_keypoints(ordered_keypoints)
            json_filename = filename.split('.')[0] + '.json'
            json_path = os.path.join(folder_path, json_filename)
            ground_truth_keypoints = load_ground_truth(json_path)
            img = cv2.imread(image_path)
            img_copy = copy.deepcopy(img)
            cv2.imwrite(os.path.join(output_path_line, filename), img_copy)
            visualize_keypoints_with_ground_truth(image_path, denormalized_keypoints, ground_truth_keypoints, output_path)
            img_with_keypoints = cv2.imread(os.path.join(output_path, filename))
            accuracy, invisible_accuracy, num_invisible = calculate_accuracy(denormalized_keypoints, ground_truth_keypoints, margin=10)
            accuracies.append(accuracy)
            invisible_accuracies.append(invisible_accuracy)
            total_inference_time.append(inference_time)
            total_invisible_keypoints += num_invisible
            print(f"Accuracy for {filename}: {accuracy}%")
            print(f"Invisible Keypoint Accuracy for {filename}: {invisible_accuracy}%")
            cv2.imwrite(os.path.join(output_path, filename), img_with_keypoints)
    overall_accuracy = np.mean(accuracies)
    overall_invisible_accuracy = np.mean(invisible_accuracies)
    avg_inference_time = np.mean(total_inference_time)
    print(f"Overall accuracy: {overall_accuracy}%")
    print(f"Overall invisible keypoint accuracy: {overall_invisible_accuracy}%")
    print(f"Total number of invisible keypoints: {total_invisible_keypoints}")
    print(f"Average inference time: {avg_inference_time}")

folder_path = '/home/jc-merlab/Pictures/Test_Data/ycb_test_03.avi'
output_path = '/home/jc-merlab/Pictures/Test_Data/vid_occ_kp_03/'
output_path_line = '/home/jc-merlab/Pictures/Data/occ_test_op_line/'

# Process video with Kalman filter applied and draw both original and smoothed keypoints:
process_video(folder_path, output_path)
# Alternatively, for processing images:
# process_folder(folder_path, output_path, output_path_line)
