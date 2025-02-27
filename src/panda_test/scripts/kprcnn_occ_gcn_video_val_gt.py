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

t = torch.cuda.get_device_properties(0).total_memory
print(t)
torch.cuda.empty_cache()

r = torch.cuda.memory_reserved(0)
print(r)
a = torch.cuda.memory_allocated(0)
print(a)
# f = r-a  # free inside reserved

weights_path = '/home/jc-merlab/Pictures/Data/trained_models/keypointsrcnn_planning_b1_e50_v8.pth'

n_nodes = 9

# to generalize home directory. User can change their parent path without entering their home directory
path = Def_Path()

# parent_path =  path.home + "/Pictures/" + "Data/"

parent_path =  "/home/schatterjee/lama/kprcnn_panda/"

# root_dir = parent_path + path.year + "-" + path.month + "-" + path.day + "/"
root_dir = parent_path + "occ_panda_physical_dataset/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.cuda.set_per_process_memory_fraction(0.9, 0)
print(device)


def train_transform():
    return A.Compose([
        A.Sequential([
            A.RandomRotate90(p=1), # Random rotation of an image by 90 degrees zero or more times
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1), # Random change of brightness & contrast
        ], p=1)
#         A.Resize(640, 480)  # Resize all images to be 640x480
    ],
    keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more at https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )

def train_test_split(src_dir):
    dst_dir_img = src_dir + "images"
    dst_dir_anno = src_dir + "annotations"
    
    if os.path.exists(dst_dir_img) and os.path.exists(dst_dir_anno):
        print("folders exist")
    else:
        os.mkdir(dst_dir_img)
        os.mkdir(dst_dir_anno)
        
    for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
        shutil.copy(jpgfile, dst_dir_img)

    for jsonfile in glob.iglob(os.path.join(src_dir, "*.json")):
        shutil.copy(jsonfile, dst_dir_anno)
        
    output = parent_path + "split_folder_output_occ_sage" + "-" + path.year + "-" + path.month + "-" + path.day

    
    splitfolders.ratio(src_dir, # The location of dataset
                   output=output, # The output location
                   seed=42, # The number of seed
                   ratio=(0.95, 0.025, 0.025), # The ratio of split dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
                   )
    
    shutil.rmtree(dst_dir_img)
    shutil.rmtree(dst_dir_anno)
    
    return output  

class KPDataset(Dataset):
    def __init__(self, root, transform=None, demo=False):                
        self.root = root
        self.transform = transform
        self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)
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
            
            # All objects are keypoints on the robot
            bboxes_labels_original = [] 
            bboxes_labels_original.append('base_joint')
            bboxes_labels_original.append('joint2')
            bboxes_labels_original.append('joint3')
            bboxes_labels_original.append('joint4')
            bboxes_labels_original.append('joint5')
            bboxes_labels_original.append('joint6')  
            bboxes_labels_original.append('joint7')
            bboxes_labels_original.append('joint8')
            bboxes_labels_original.append('joint9')

        if self.transform:   
            # Converting keypoints from [x,y,visibility]-format to [x, y]-format + Flattening nested list of keypoints            
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]], where each keypoint is in [x, y]-format            
            # Then we need to convert it to the following list:
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2]
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]
            
            # Apply augmentations
            transformed = self.transform(image=img_original, bboxes=bboxes_original, bboxes_labels=bboxes_labels_original, keypoints=keypoints_original_flattened)
            img = transformed['image']
            bboxes = transformed['bboxes']
            # Unflattening list transformed['keypoints']
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]]
            keypoints_transformed_unflattened = np.reshape(np.array(transformed['keypoints']), (-1,1,2)).tolist()

            # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
            keypoints = []
            for o_idx, obj in enumerate(keypoints_transformed_unflattened):
#                 print("object", obj)
#                 print(" obj index", o_idx)# Iterating over objects
                obj_keypoints = []
                for k_idx, kp in enumerate(obj): # Iterating over keypoints in each object
                    obj_keypoints.append(kp + [keypoints_original[o_idx][k_idx][2]])
                keypoints.append(obj_keypoints)
        
        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original        
        
        # Convert everything into a torch tensor        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)       
        target = {}
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]  
#         labels = [1, 2, 3, 4, 5, 6]
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64) # all objects are joint positions
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)
        img = F.to_tensor(img)        
        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor(labels, dtype=torch.int64) # all objects are glue tubes
        target_original["image_id"] = torch.tensor([idx])
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (bboxes_original[:, 2] - bboxes_original[:, 0])
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
#     print(f"keypoints_gt shape: {keypoints_gt.shape}")  # Debug print
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
#     print("ground truth dist and angles", distances_angles)
    return distances_angles

class KeypointPipeline(nn.Module):
    def __init__(self, weights_path):
        super(KeypointPipeline, self).__init__()  
        self.keypoint_model = torch.load(weights_path).to(device)
        self.graph_gcn = GraphGCN(8,1024,4)
        
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
        
#         print("initial predicted keypoints", keypoints)
        
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

def kgnn2d_loss(gt_keypoints, pred_keypoints, gt_distances_next, gt_angles_next, gt_distances_prev, gt_angles_prev, pred_distances_next, pred_angles_next, pred_distances_prev, pred_angles_prev):
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
        denormalized_x = (kp[:, 0] * (width / 2)) + (width / 2)
        denormalized_y = (kp[:, 1] * (height / 2)) + (height / 2)
        denormalized_kp = torch.stack((denormalized_x, denormalized_y), dim=1)
        denormalized_keypoints.append(denormalized_kp)
    denormalized_keypoints = torch.stack(denormalized_keypoints)
    return denormalized_keypoints

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
checkpoint_path = '/home/jc-merlab/Pictures/Data/trained_models/gcn_ckpt/kprcnn_occ_gcn_ckpt_b128e17.pth'
checkpoint = torch.load(checkpoint_path)

# Extract the state dictionary
model_state_dict = checkpoint['model_state_dict']

# Load the state dictionary into the model
model.load_state_dict(model_state_dict)

model.eval()  # Set the model to evaluation mode

def prepare_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img).to(device)
    return img_tensor

def load_ground_truth(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # ground_truth_keypoints = [[int(kp[0][0]), int(kp[0][1])] for kp in data['keypoints']]
    ground_truth_keypoints = [[int(kp[0][0]), int(kp[0][1]), kp[0][2]] for kp in data['keypoints']]
    return ground_truth_keypoints

def draw_lines_between_keypoints(image, keypoints, color=(255, 255, 255)):
    for i in range(len(keypoints) - 1):
        start_point = (int(keypoints[i][0]), int(keypoints[i][1]))
        end_point = (int(keypoints[i + 1][0]), int(keypoints[i + 1][1]))
        cv2.line(image, start_point, end_point, color, thickness=2)
        
    for x, y in keypoints:
        cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

def predict(model, img_tensor):
    with torch.no_grad():
        KGNN2D = model([img_tensor])

    # print("Predicted keypoints", KGNN2D)
    return KGNN2D

def postprocess_keypoints(keypoints, width=640, height=480):
    denormalized_keypoints = denormalize_keypoints(keypoints, width, height)
    return denormalized_keypoints

def visualize_keypoints_with_ground_truth(image_path, predicted_keypoints, ground_truth_keypoints, out_dir):
    img = cv2.imread(image_path)
    if torch.is_tensor(predicted_keypoints):
        predicted_keypoints = predicted_keypoints.cpu().numpy()
    for kp in predicted_keypoints[0]:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(img, (x, y), radius=8, color=(255, 0, 0), thickness=-1)
    
    for x, y, _ in ground_truth_keypoints:
        cv2.circle(img, (x, y), radius=6, color=(0, 0, 255), thickness=-1)
    
    filename = os.path.basename(image_path)
    output_path = os.path.join(out_dir, filename)
    cv2.imwrite(output_path, img)

def visualize_keypoints(image, predicted_keypoints, gt_keypoints, out_dir, frame_id):
    img = image.copy()  # Use the frame directly
    if torch.is_tensor(predicted_keypoints):
        predicted_keypoints = predicted_keypoints.cpu().numpy()
    if torch.is_tensor(gt_keypoints):
        gt_keypoints = gt_keypoints.cpu().numpy()

    # Draw GT keypoints (always present, marked in green)
    for kp in gt_keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(img, (x, y), radius=10, color=(0, 255, 0), thickness=-1)  # Green for GT

    # Draw predicted keypoints (ignoring NaN values, marked in red)
    for kp in predicted_keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(img, (x, y), radius=7, color=(0, 0, 255), thickness=-1)  # Red for predicted
    
    output_path = os.path.join(out_dir, f"frame_{frame_id:04d}.jpg")  # Save with frame index
    cv2.imwrite(output_path, img)

def calculate_accuracy(predicted_keypoints, ground_truth_keypoints, margin=10):
    """
    Calculate the accuracy of predicted keypoints within a margin of 10 pixels.
    Also calculate accuracy for invisible keypoints within a margin of 5 pixels.
    """
    correct = 0
    total = len(ground_truth_keypoints)
    
    correct_invisible = 0
    total_invisible = 0

    for pred_kp, gt_kp in zip(predicted_keypoints[0], ground_truth_keypoints):
        pred_kp = pred_kp.cpu().numpy()  # Move tensor to CPU and convert to NumPy array
        dist = np.linalg.norm(np.array(pred_kp[:2]) - np.array(gt_kp[:2]))  # Use only x, y for distance calculation

        if gt_kp[2] == 0:  # Invisible keypoint
            total_invisible += 1
            if dist <= margin:  # Margin for invisible keypoints
                correct_invisible += 1
        else:  # Visible keypoint
            if dist <= margin:
                correct += 1
    
    correct_total = correct + correct_invisible
    accuracy = (correct_total / total) * 100
    invisible_accuracy = (correct_invisible / total_invisible) * 100 if total_invisible > 0 else 0
    return accuracy, invisible_accuracy, total_invisible

def extract_predicted_keypoints(folder_path, model, device, output_path, gt_keypoints):
    cap = cv2.VideoCapture(folder_path)
    frame_id = 0
    predicted_keypoints = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(frame)    
        img_tensor = F.to_tensor(image).to(device)
        # image.unsqueeze_(0)
        # img_tensor = list(image)
        KGNN2D = predict(model, img_tensor)
        ordered_keypoints = reorder_batch_keypoints(KGNN2D)
        denormalized_keypoints = postprocess_keypoints(ordered_keypoints).detach().cpu().numpy().squeeze(0)
        visualize_keypoints(frame, denormalized_keypoints, gt_keypoints[frame_id], output_path, frame_id)

        predicted_keypoints.append(np.array(denormalized_keypoints))
    
        frame_id += 1

    cap.release()
 
    # Closes all the frames
    cv2.destroyAllWindows()

    return predicted_keypoints

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

        # Apply the same y-axis limit for consistency
        ax.set_ylim(0, 150)

    plt.tight_layout()
    plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

weights_path = '/home/jc-merlab/Pictures/Data/trained_models/keypointsrcnn_planning_b1_e50_v8.pth'
model_kprcnn = torch.load(weights_path).to(device)
model_kprcnn.eval()

# **Final Execution**
non_occluded_video_path = "/home/jc-merlab/Pictures/Test_Data/vid_occ_gt/exp_03/output_video.avi"
gt_keypoints = extract_gt_keypoints(non_occluded_video_path, model_kprcnn, device)


print(np.array(gt_keypoints).shape)

model_kpgcn = KeypointPipeline(weights_path)
model_kpgcn = model_kpgcn.to(device)

occluded_video_path = "/home/jc-merlab/Pictures/Test_Data/vid_occ_pred/exp_03/output_video.avi"
output_path = '/home/jc-merlab/Pictures/Test_Data/vid_occ_kpgcn_gt_03/'
os.makedirs(output_path, exist_ok=True)

# Load the checkpoint
checkpoint_path = '/home/jc-merlab/Pictures/Data/trained_models/gcn_ckpt/kprcnn_occ_gcn_ckpt_b128e17.pth'
checkpoint = torch.load(checkpoint_path)

# Extract the state dictionary
model_state_dict = checkpoint['model_state_dict']

# Load the state dictionary into the model
model_kpgcn.load_state_dict(model_state_dict)

model_kpgcn.eval()  # Set the model to evaluation mode

pred_keypoints = extract_predicted_keypoints(occluded_video_path, model_kpgcn, device, output_path, gt_keypoints)

print(np.array(pred_keypoints).shape)

errors_per_frame, errors_per_keypoint = compute_errors(gt_keypoints, pred_keypoints)
visualize_errors(errors_per_frame, errors_per_keypoint)
visualize_individual_keypoint_errors(errors_per_keypoint)









































































