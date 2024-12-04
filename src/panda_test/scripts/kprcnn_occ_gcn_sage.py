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
import random

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

# parent_path =  "/home/jc-merlab/Pictures/Data/"

parent_path =  "/home/schatterjee/lama/kprcnn_panda/"

# root_dir = parent_path + path.year + "-" + path.month + "-" + path.day + "/"
root_dir = parent_path + "occ_new_panda_physical_dataset/"


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
        
    output = parent_path + "split_folder_output_occ_gcn_sage_v1" + "-" + path.year + "-" + path.month + "-" + path.day 
    
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
#     print("ground truth dist and angles", distances_angles)
    return distances_angles

def visualize_keypoints(image, pred_keypoints, gt_keypoints=None, title="Keypoints Visualization"):
    # Convert the image tensor to a numpy array and move channels (C, H, W) -> (H, W, C)
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    
    # Plot predicted keypoints (blue)
    if pred_keypoints is not None:
        pred_keypoints_np = pred_keypoints.cpu().numpy()
        for kp in pred_keypoints_np:
            plt.scatter(kp[0], kp[1], c='blue', marker='x', s=100, label='Predicted' if kp is pred_keypoints_np[0] else "")
    
    # Plot ground truth keypoints (green)
    if gt_keypoints is not None:
        gt_keypoints_np = gt_keypoints.cpu().numpy()
        for kp in gt_keypoints_np:
            plt.scatter(kp[0], kp[1], c='green', marker='o', s=100, label='Ground Truth' if kp is gt_keypoints_np[0] else "")
    
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()

class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_1, hidden_channels_2,  out_channels):
        super(GraphGCN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels_1)
        self.conv2 = GCNConv(hidden_channels_1, hidden_channels_1)
        self.conv3 = SAGEConv(hidden_channels_1, hidden_channels_2)
        self.conv4 = GCNConv(hidden_channels_2, hidden_channels_2)
        self.fc = torch.nn.Linear(hidden_channels_2, out_channels)

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
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.conv4(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.fc(x)
        return x


class KeypointPipeline(nn.Module):
    def __init__(self, weights_path):
        super(KeypointPipeline, self).__init__()
        self.keypoint_model = torch.load(weights_path).to(device)
        self.graph_gcn = GraphGCN(8,1024,512,4)
        
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

    
    def apply_transform(self, img, pred_keypoints, gt_keypoints=None):
        img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert (C, H, W) -> (H, W, C)

        # Prepare keypoints for transformation with all attributes (x, y, confidence, label)
        pred_keypoints_np = [kp[:2].tolist() for kp in pred_keypoints]
        pred_attributes = [kp[2:].tolist() for kp in pred_keypoints]

        pred_keypoints_np = pred_keypoints[:, :2].cpu().numpy().tolist()

        gt_keypoints_np = [kp[:2].tolist() for kp in gt_keypoints] if gt_keypoints is not None else []

        # Get the original image height and width
        orig_height, orig_width = img_np.shape[:2]

        # Clamp keypoints to stay within the original image bounds BEFORE transformation
        def clamp_keypoints(keypoints, max_width, max_height):
            clamped_kps = []
            for kp in keypoints:
                x, y = kp[:2]
                # Clamp x and y within the image boundaries (max_width-1, max_height-1)
                x_clamped = min(max(x, 0), max_width - 1)  # x should be within [0, max_width-1]
                y_clamped = min(max(y, 0), max_height - 1)  # y should be within [0, max_height-1]
                clamped_kps.append([x_clamped, y_clamped])
            return clamped_kps

        # Clamp the predicted keypoints to the original image bounds
        pred_keypoints_np = clamp_keypoints(pred_keypoints_np, orig_width, orig_height)

        # Clamp the ground truth keypoints to the original image bounds if they exist
        if gt_keypoints_np:
            gt_keypoints_np = clamp_keypoints(gt_keypoints_np, orig_width, orig_height)

        post_prediction_transform = A.Compose(
            [
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.2),
                A.VerticalFlip(p=0.2),
                A.OneOf([
                    A.Resize(height=random.randint(240, 720), width=random.randint(320, 1280)),
                    A.NoOp()  # No operation (keeps original size)
                ], p=0.5)  # 50% chance to resize
            ],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=True)
        )

        # Apply the transformation after clamping
        transformed = post_prediction_transform(
            image=img_np, keypoints=pred_keypoints_np + gt_keypoints_np
        )

        transformed_img = F.to_tensor(transformed['image']).to(device)

        # Get the new image size after transformation
        new_height, new_width = transformed_img.shape[1], transformed_img.shape[2]

        # Separate transformed predicted and ground truth keypoints
        pred_kps_transformed = transformed['keypoints'][:len(pred_keypoints_np)]
        gt_kps_transformed = transformed['keypoints'][len(pred_keypoints_np):] if gt_keypoints_np else None

        # Add back the confidence and labels to the transformed predicted keypoints
        pred_kps_transformed = [
            list(pred_kp) + pred_attr for pred_kp, pred_attr in zip(pred_kps_transformed, pred_attributes)
        ]

        # Fix the warning by using .clone().detach()
        pred_kps_transformed = torch.tensor(pred_kps_transformed, dtype=torch.float32).clone().detach().to(device)

        # Convert ground truth keypoints to a tensor
        gt_kps_transformed_tensor = (
            torch.tensor(gt_kps_transformed, dtype=torch.float32).clone().detach().to(device) if gt_kps_transformed else None
        )

        return transformed_img, pred_kps_transformed, gt_kps_transformed_tensor
    
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
                
                prev_kp = keypoints_dict.get(prev_label, None)
                next_kp = keypoints_dict.get(next_label, None)
                
#                 print("Previous Kp", prev_kp)
#                 print("Next Kp", next_kp)

                if prev_kp is None and next_kp is None:
                    avg_x, avg_y = image_width / 2, image_height / 2
                elif prev_kp is None:
                    avg_x, avg_y = (next_kp[0] + image_width / 2) / 2, (next_kp[1] + image_height / 2) / 2
                elif next_kp is None:
                    avg_x, avg_y = (prev_kp[0] + image_width / 2) / 2, (prev_kp[1] + image_height / 2) / 2
                else:
                    avg_x = (prev_kp[0] + next_kp[0]) / 2
                    avg_y = (prev_kp[1] + next_kp[1]) / 2

                complete_keypoints.append([avg_x, avg_y, 0, i])
#                 print(f"Filled missing keypoint for label {i} at position ({avg_x}, {avg_y})")

                
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
        
#         print(Data(x=node_features, edge_index=edge_index))
        return Data(x=node_features, edge_index=edge_index)
    
    def forward(self, imgs, gt_keypoints_batch=None):
        keypoint_model_training = self.keypoint_model.training
        self.keypoint_model.eval()

        with torch.no_grad():
            batch_outputs = [self.keypoint_model(img.unsqueeze(0).to(device)) for img in imgs]

        self.keypoint_model.train(mode=keypoint_model_training)

        batch_pred_keypoints = [self.process_model_output(output) for output in batch_outputs]
        
        if gt_keypoints_batch is not None:  # If ground truth is provided (Training Mode)
            batch_transformed_data = []
            for img, pred_kp, gt_kp in zip(imgs, batch_pred_keypoints, gt_keypoints_batch):
                transformed_img, pred_kp_transformed, gt_kp_transformed = \
                    self.apply_transform(img, pred_kp, gt_kp)
#                 print("Pred KP transformed: ", pred_kp_transformed)
                # Call the visualization function
                # visualize_keypoints(transformed_img, pred_kp_transformed, gt_kp_transformed, \
                #                  title="Transformed Keypoints")
                pred_kp_filled = self.fill_missing_keypoints(pred_kp_transformed, transformed_img.shape[1], transformed_img.shape[0])
                pred_kp_normalized = self.normalize_keypoints(pred_kp_filled, transformed_img.shape[1], transformed_img.shape[0])

                batch_transformed_data.append((pred_kp_filled[:, :2], pred_kp_normalized, gt_kp_transformed))

            all_graphs = [self.keypoints_to_graph(kp, transformed_img.shape[1], transformed_img.shape[0]) \
                          for _, kp, _ in batch_transformed_data]
            all_predictions = [self.graph_gcn(graph.x, graph.edge_index) for graph in all_graphs]

            return torch.stack(all_predictions), [gt_kp for _, _, gt_kp in batch_transformed_data], [init_kp for init_kp, _, _ in batch_transformed_data], transformed_img.shape[1], transformed_img.shape[0]
        
        else:  # Inference Mode
            batch_final_keypoints = []
            for pred_kp in batch_pred_keypoints:
                pred_kp_filled = self.fill_missing_keypoints(pred_kp, 640, 480)
                pred_kp_normalized = self.normalize_keypoints(pred_kp_filled, 640, 480)
                batch_final_keypoints.append(pred_kp_normalized)

            return batch_final_keypoints  # Final keypoints for inference


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
#         print("predicted normalized keypoints for reordering")
#         print(normalized_keypoints)
        reordered_normalized_keypoints = torch.zeros(num_keypoints, 2, device=batch_keypoints.device)
        rounded_labels = torch.round(normalized_keypoints[:, 3]).int()
#         print("rounded labels", rounded_labels)
        used_indices = []
        for label in range(1, 10):
#             print("label", label)
            valid_idx = (rounded_labels == label).nonzero(as_tuple=True)[0]
#             print("valid index", valid_idx.numel())
            if valid_idx.numel() > 0:
#                 print("index in valid index is present")
                reordered_normalized_keypoints[label - 1] = normalized_keypoints[valid_idx[0], :2]
            else:
#                 print("used_indices")
#                 print("index in valid index is not present")
                invalid_idx = ((rounded_labels < 1) | (rounded_labels > 9)).nonzero(as_tuple=True)[0]
#                 print("invalid idx ", invalid_idx)
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

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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

checkpoint_dir = '/home/schatterjee/lama/kprcnn_panda/trained_models/gcn_sage_ckpt_v1/'
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

start_time = time.time()

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0

    for imgs, target_dicts, _ in data_loader_train:
        imgs = [img.to(device) for img in imgs]
        optimizer.zero_grad()
        gt_keypoints_batch = process_batch_keypoints(target_dicts)

        with autocast():
            KGNN2D, transformed_gt_keypoints, init_kp_denorm, width, height = model(imgs, gt_keypoints_batch)
            reordered_normalized_keypoints = reorder_batch_keypoints(KGNN2D)
            denormalized_keypoints = denormalize_keypoints(reordered_normalized_keypoints, width, height)
            transformed_gt_keypoints = torch.stack(transformed_gt_keypoints)
            init_kp_denorm = torch.stack(init_kp_denorm)
            gt_distances_angles = calculate_gt_distances_angles(transformed_gt_keypoints)
            init_distances_angles = calculate_gt_distances_angles(init_kp_denorm)
            
            pred_distances_angles = calculate_gt_distances_angles(denormalized_keypoints)
            loss_kprcnn = kgnn2d_loss(transformed_gt_keypoints, init_kp_denorm, gt_distances_angles[:, :, 0],
                                      gt_distances_angles[:, :, 1], gt_distances_angles[:, :, 2],
                                      gt_distances_angles[:, :, 3], init_distances_angles[:, :, 0],
                                      init_distances_angles[:, :, 1], init_distances_angles[:, :, 2],
                                      init_distances_angles[:, :, 3])
            loss_kgnn2d = kgnn2d_loss(transformed_gt_keypoints, denormalized_keypoints, gt_distances_angles[:, :, 0],
                                      gt_distances_angles[:, :, 1], gt_distances_angles[:, :, 2],
                                      gt_distances_angles[:, :, 3], pred_distances_angles[:, :, 0],
                                      pred_distances_angles[:, :, 1], pred_distances_angles[:, :, 2],
                                      pred_distances_angles[:, :, 3])
            
            final_loss = loss_kprcnn + loss_kgnn2d

        scaler.scale(final_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += final_loss.item()
    
    # Calculate the epoch duration
    epoch_time = time.time() - start_time  # Elapsed time for the epoch
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader_train)}, Time: {epoch_time:.2f} seconds')


# Save checkpoint every epoch
    if (epoch + 1) % 1 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'kprcnn_occ_gcn_sage_ckpt_v1_b{batch_size}e{epoch+1}.pth')
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


# Save final model
model_save_path = f"/home/schatterjee/lama/kprcnn_panda/trained_models/kprcnn_gcn_sage_v1_b{batch_size}_e{num_epochs}.pth"
torch.save(model.state_dict(), model_save_path)


