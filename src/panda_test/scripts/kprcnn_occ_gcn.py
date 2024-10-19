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

import transforms, utils, engine, train
from utils import collate_fn
from engine import train_one_epoch, evaluate

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

weights_path = '/home/schatterjee/lama/kprcnn_panda/trained_models/keypointsrcnn_planning_b1_e50_v8.pth'

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
        
    output = parent_path + "split_folder_output_occ_gcn" + "-" + path.year + "-" + path.month + "-" + path.day

    
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
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
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

def post_prediction_transform():
    return A.Compose([
        A.Resize(320, 240),  # Resize image and keypoints
        A.Rotate(limit=30, p=0.5),  # Random rotation up to 30 degrees
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True),
       bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

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
    
    
    def apply_post_transform(self, img, keypoints, bboxes, labels):
        # Convert tensors to numpy for transformation
        img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
        keypoints_np = [kp[:2].tolist() for kp in keypoints]  # Extract (x, y) from keypoints
        bboxes_np = bboxes.cpu().numpy()

        # Apply transformations
        transformed = self.post_transform(image=img_np, keypoints=keypoints_np, bboxes=bboxes_np, labels=labels)

        # Convert back to tensor
        transformed_img = F.to_tensor(transformed['image']).to(device)
        transformed_keypoints = torch.tensor(transformed['keypoints'], dtype=torch.float32).to(device)
        transformed_bboxes = torch.tensor(transformed['bboxes'], dtype=torch.float32).to(device)

        return transformed_img, transformed_keypoints, transformed_bboxes
    
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

        batch_transformed_data = []
        
        for img, output in zip(imgs, batch_outputs):
            keypoints = self.process_model_output(output)
            bboxes = output[0]['boxes'].detach()
            labels = output[0]['labels'].detach().cpu().tolist()

            # Apply random post-prediction transformations
            transformed_img, transformed_keypoints, transformed_bboxes = self.apply_post_transform(
                img, keypoints, bboxes, labels
            )

            # Extract the actual image dimensions dynamically
            height, width = transformed_img.shape[1], transformed_img.shape[2]

            batch_transformed_data.append((transformed_img, transformed_keypoints, transformed_bboxes, width, height))

            # Use the extracted dimensions for graph creation
            all_graphs = [
                self.keypoints_to_graph(keypoints, width, height) 
                for _, keypoints, _, width, height in batch_transformed_data
            ]

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
                invalid_idx = ((rounded_labels < 1) | (rounded_labels > 6)).nonzero(as_tuple=True)[0]
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

model = KeypointPipeline(weights_path)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()

num_epochs = 100
batch_size = 128

split_folder_path = train_test_split(root_dir)
KEYPOINTS_FOLDER_TRAIN = split_folder_path +"/train"
KEYPOINTS_FOLDER_VAL = split_folder_path +"/val"
KEYPOINTS_FOLDER_TEST = split_folder_path +"/test"

dataset_train = KPDataset(KEYPOINTS_FOLDER_TRAIN, transform=None, demo=False)
dataset_val = KPDataset(KEYPOINTS_FOLDER_VAL, transform=None, demo=False)
dataset_test = KPDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)

data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

checkpoint_dir = '/home/schatterjee/lama/kprcnn_panda/trained_models/gcn_ckpt/'
checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')

# Create checkpoint directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

# Load checkpoint if exists
start_epoch = 0
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from epoch {start_epoch}")


for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0

    for imgs, target_dicts, _ in data_loader_train:
        imgs = [img.to(device) for img in imgs]
        optimizer.zero_grad()
        
        with autocast():
            KGNN2D = model(imgs)            
            keypoints_gt = process_batch_keypoints(target_dicts)
            # print("Ground truth keypoints", keypoints_gt)
            reordered_normalized_keypoints = reorder_batch_keypoints(KGNN2D)
            denormalized_keypoints = denormalize_keypoints(reordered_normalized_keypoints)
            # print("Final precicted keypoints", denormalized_keypoints)
            gt_distances_angles = calculate_gt_distances_angles(keypoints_gt)
            pred_distances_angles = calculate_gt_distances_angles(denormalized_keypoints)
            loss_kgnn2d = kgnn2d_loss(keypoints_gt, denormalized_keypoints, gt_distances_angles[:, :, 0], 
                                      gt_distances_angles[:, :, 1], gt_distances_angles[:, :, 2], 
                                      gt_distances_angles[:, :, 3], pred_distances_angles[:, :, 0], 
                                      pred_distances_angles[:, :, 1], pred_distances_angles[:, :, 2], 
                                      pred_distances_angles[:, :, 3])
            
        scaler.scale(loss_kgnn2d).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss_kgnn2d.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(data_loader_train)}')
    
# Save checkpoint every epoch
    if (epoch + 1) % 1 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'kprcnn_occ_gcn_ckpt_b{batch_size}e{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')

    # Save latest checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }, checkpoint_path)

# end_time = time.time()

# total_time = end_time - start_time
# print("total time", total_time)

# Save final model
model_save_path = f"/home/schatterjee/lama/kprcnn_panda/trained_models/kprcnn_gcn_b{batch_size}_e{num_epochs}.pth"
torch.save(model.state_dict(), model_save_path)







