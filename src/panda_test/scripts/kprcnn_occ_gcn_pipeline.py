import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GCNConv
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

import albumentations as A # Library for augmentations

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_nodes = 9

def calculate_distance_angle(kp1, kp2):
    dx = kp2[0] - kp1[0]
    dy = kp2[1] - kp1[1]
    dx, dy = torch.tensor(dx).to(device), torch.tensor(dy).to(device)
    distance = torch.sqrt(dx ** 2 + dy ** 2)
    angle = torch.atan2(dy, dx)
    return distance, angle

def calculate_gt_distances_angles(keypoints_gt):
#     print(f"keypoints_gt shape: {keypoints_gt.shape}")  # Debug print
    print(keypoints_gt.shape)
    batch_size, num_keypoints, num_dims = keypoints_gt.shape    
    assert num_keypoints == n_nodes and num_dims == 2, "keypoints_gt must have shape (batch_size, 9, 2)"
    distances_angles = []

    for b in range(batch_size):
        batch_distances_angles = torch.zeros((num_keypoints, 4), dtype=torch.float32).to(device)  # Initialize with zeros

        for i in range(num_keypoints):
            current_kp = keypoints_gt[b, i]
            next_i = (i + 1) % num_keypoints
            prev_i = (i - 1 + num_keypoints) % num_keypoints

            # Calculate distance and angle to the next keypoint
            dist, angle = calculate_distance_angle(current_kp, keypoints_gt[b, next_i])
            batch_distances_angles[i, 0] = dist
            batch_distances_angles[i, 1] = angle

            # Calculate distance and angle to the previous keypoint
            dist, angle = calculate_distance_angle(current_kp, keypoints_gt[b, prev_i])
            batch_distances_angles[i, 2] = dist
            batch_distances_angles[i, 3] = angle

        distances_angles.append(batch_distances_angles)

    distances_angles = torch.stack(distances_angles)
    return distances_angles

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

class KeypointPipeline(nn.Module):
    def __init__(self, weights_path):
        super(KeypointPipeline, self).__init__()  
        self.keypoint_model = torch.load(weights_path).to(device)
        self.graph_gcn = GraphGCN(8,512,4)
        
    def process_model_output(self, output):
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.01)[0].tolist()
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()
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
                
        return keypoints
    
    def fill_missing_keypoints(self, keypoints, image_width, image_height):
        keypoints_dict = {int(kp[3]): kp for kp in keypoints}
        complete_keypoints = []
        labels = [int(kp[3]) for kp in keypoints]
        
        # Identify missing labels
        all_labels = set(range(1, 10))
        missing_labels = list(all_labels - set(labels))
        missing_labels.sort()
        
        # Handle consecutive missing labels by placing them at the midpoint or image center
        for i in range(1, 10):
            if i in keypoints_dict:
                complete_keypoints.append(keypoints_dict[i].tolist())
            else:
                prev_label = i - 1 if i > 1 else 9
                next_label = i + 1 if i < 9 else 1
                prev_kp = keypoints_dict.get(prev_label, [image_width / 2, image_height / 2, 0, prev_label])
                next_kp = keypoints_dict.get(next_label, [image_width / 2, image_height / 2, 0, next_label])
                
                if next_label in missing_labels:
                    next_kp = [image_width / 2, image_height / 2, 0, next_label]
                avg_x = (prev_kp[0] + next_kp[0]) / 2
                avg_y = (prev_kp[1] + next_kp[1]) / 2
                complete_keypoints.append([avg_x, avg_y, 0, i])
                
#         print("filled missing keypoints", complete_keypoints)
        
        return torch.tensor(complete_keypoints, dtype=torch.float32).to(device)

    def normalize_keypoints(self, keypoints, image_width, image_height):
        keypoints[:, 0] = (keypoints[:, 0] - image_width / 2) / (image_width / 2)
        keypoints[:, 1] = (keypoints[:, 1] - image_height / 2) / (image_height / 2)
        return keypoints
    

    def keypoints_to_graph(self, keypoints, image_width, image_height):
        node_features = []
        for i, kp in enumerate(keypoints):
            x, y, conf, label = kp
            prev_kp = keypoints[i - 1]
            next_kp = keypoints[(i + 1) % len(keypoints)]
            dist_next, angle_next = calculate_distance_angle([x, y], next_kp[:2])
            dist_prev, angle_prev = calculate_distance_angle([x, y], prev_kp[:2])
            node_features.append([x, y, conf, label, dist_next, angle_next, dist_prev, angle_prev])
        node_features = torch.tensor(node_features, dtype=torch.float32).to(device)
        edge_index = torch.tensor([[i, (i + 1) % len(keypoints)] for i in range(len(keypoints))] + 
                                  [[(i + 1) % len(keypoints), i] for i in range(len(keypoints))], dtype=torch.long).t().contiguous().to(device)
        return Data(x=node_features, edge_index=edge_index)
    
    def forward(self, imgs):
        keypoint_model_training = self.keypoint_model.training
        self.keypoint_model.eval()

        with torch.no_grad():
            batch_outputs = [self.keypoint_model(img.unsqueeze(0).to(device)) for img in imgs]

        self.keypoint_model.train(mode=keypoint_model_training)

        batch_labeled_keypoints = [self.process_model_output(output) for output in batch_outputs]

        # Fill missing keypoints first and then normalize them
        for idx, keypoints in enumerate(batch_labeled_keypoints):
            image_width, image_height = 640, 480
            filled_keypoints = self.fill_missing_keypoints(keypoints, image_width, image_height)
            normalized_keypoints = self.normalize_keypoints(filled_keypoints, image_width, image_height)
            batch_labeled_keypoints[idx] = normalized_keypoints
       
        all_graphs = [self.keypoints_to_graph(keypoints, 640, 480) for keypoints in batch_labeled_keypoints]
        all_predictions = [self.graph_gcn(graph.x, graph.edge_index) for graph in all_graphs]

        final_predictions = torch.stack(all_predictions)

        return final_predictions

# Utility function to load the model
def load_keypoint_pipeline(weights_path):
    model = KeypointPipeline(weights_path)
    model.to(device)
    return model


