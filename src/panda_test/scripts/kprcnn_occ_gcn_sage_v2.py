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
from utils import collate_fn

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

from kprcnn_occ_gcn_pipeline import load_keypoint_pipeline

import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from torch.autograd import Variable
import torch_geometric.nn as pyg
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch_geometric.data import Data


path = Def_Path()

t = torch.cuda.get_device_properties(0).total_memory
print(t)
torch.cuda.empty_cache()

r = torch.cuda.memory_reserved(0)
print(r)
a = torch.cuda.memory_allocated(0)
print(a)
# f = r-a  # free inside reserved

weights_path = '/home/schatterjee/lama/kprcnn_panda/trained_models/keypointsrcnn_planning_b1_e50_v8.pth'
# weights_path = '/home/jc-merlab/Pictures/Data/trained_models/keypointsrcnn_planning_b1_e50_v8.pth'

n_nodes = 9
# to generalize home directory. User can change their parent path without entering their home directory

parent_path =  path.home + "/Pictures/" + "Data/"
# parent_path = "/home/jc-merlab/Pictures/Data/"
parent_path =  "/home/schatterjee/lama/kprcnn_panda/"

root_dir = parent_path + "occ_new_panda_physical_dataset/"
# root_dir = parent_path + "occ_new_panda_physical_dataset_trunc_v2/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.cuda.set_per_process_memory_fraction(0.9, 0)
print(device)

# Load the checkpoint
checkpoint_path = '/home/schatterjee/lama/kprcnn_panda/trained_models/gcn_ckpt_v2/kprcnn_occ_gcn_ckpt_b128e23.pth'
checkpoint = torch.load(checkpoint_path)

# Extract the state dictionary
model_state_dict = checkpoint['model_state_dict']

# # Load the state dictionary into the model

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
        
    output = parent_path + "split_folder_output_gcn_sage_v2" + "-" + path.year + "-" + path.month + "-" + path.day 
    
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

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader

class GTDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))

    def __len__(self):
        return len(self.imgs_files)

    def __getitem__(self, idx):
        img_file = self.imgs_files[idx]
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        annotations_path = os.path.join(self.root, "annotations", self.annotations_files[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        with open(annotations_path) as f:
            data = json.load(f)
            bboxes = data['bboxes']
            keypoints = data['keypoints']
            
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)       
        target = {}
        target["boxes"] = bboxes
        target["image_id"] = torch.tensor([idx])
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)
        img = F.to_tensor(img)            
        
        return img, target, img_file

def calculate_distance_angle(kp1, kp2):
    dx = kp2[0] - kp1[0]
    dy = kp2[1] - kp1[1]
    dx, dy = torch.tensor(dx).to(device), torch.tensor(dy).to(device)
    distance = torch.sqrt(dx ** 2 + dy ** 2)
    angle = torch.atan2(dy, dx)
    return distance, angle

def calculate_gt_distances_angles(keypoints_gt):
#     print(f"keypoints_gt shape: {keypoints_gt.shape}")  # Debug print
#     print(keypoints_gt.shape)
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

def denormalize_keypoints(batch_keypoints, width=640, height=480):
    denormalized_keypoints = []
    for kp in batch_keypoints:
        denormalized_x = (kp[:, 0] * (width / 2)) + (width / 2)
        denormalized_y = (kp[:, 1] * (height / 2)) + (height / 2)
        denormalized_kp = torch.stack((denormalized_x, denormalized_y), dim=1)
        denormalized_keypoints.append(denormalized_kp)
    denormalized_keypoints = torch.stack(denormalized_keypoints)
    return denormalized_keypoints

def process_batch_keypoints(target_dicts):
    keypoints_batch = target_dicts['keypoints']  # Shape: (batch_size, num_keypoints, 1, 3)
    keypoints_gt = keypoints_batch.squeeze(2)[:, :, :2]  # Shape: (batch_size, num_keypoints, 2)
    
    return keypoints_gt.to(device)

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

class KeypointV2Pipeline(nn.Module):
    def __init__(self, weights_path, model_state_dict):
        super(KeypointV2Pipeline, self).__init__()
        self.keypoint_model = model = load_keypoint_pipeline(weights_path)
        self.keypoint_model.load_state_dict(model_state_dict)
        self.graph_gcn = GraphGCN(6,512,256,2)
       
    def process_model_output(self, output):
#         print("Original output:", output)

        # Denormalize only the x and y coordinates
        denormalized_keypoints = denormalize_keypoints(output)  # (batch_size, keypoints, 2)
#         print("Denormalized Keypoints (x, y) Shape:", denormalized_keypoints)
        return denormalized_keypoints
    
    def apply_transform(self, img, pred_keypoints, gt_keypoints=None):
#         print("Predicted keypoints for clamping", pred_keypoints)
        img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert (C, H, W) -> (H, W, C)
        # Prepare keypoints for transformation with all attributes (x, y, confidence, label)
        
        pred_keypoints_np = pred_keypoints[0].cpu().numpy().tolist()
        

        gt_keypoints_np = [kp[:2].tolist() for kp in gt_keypoints] if gt_keypoints is not None else []

        # Get the original image height and width
        orig_height, orig_width = img_np.shape[:2]

        # Clamp keypoints to stay within the original image bounds BEFORE transformation
        def clamp_keypoints(keypoints, max_width, max_height):            
            clamped_kps = []
            for kp in keypoints:
                x, y = kp
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


        # Fix the warning by using .clone().detach()
        pred_kps_transformed = torch.tensor(pred_kps_transformed, dtype=torch.float32).clone().detach().to(device)

        # Convert ground truth keypoints to a tensor
        gt_kps_transformed_tensor = (
            torch.tensor(gt_kps_transformed, dtype=torch.float32).clone().detach().to(device) if gt_kps_transformed else None
        )

        return transformed_img, pred_kps_transformed, gt_kps_transformed_tensor
    
    def normalize_keypoints(self, keypoints, image_width, image_height):
        keypoints[:, 0] = (keypoints[:, 0] - image_width / 2) / (image_width / 2)
        keypoints[:, 1] = (keypoints[:, 1] - image_height / 2) / (image_height / 2)
        return keypoints    

    def keypoints_to_graph(self, keypoints, image_width, image_height):
        node_features = []
        for i, kp in enumerate(keypoints):
            x, y = kp
            prev_kp = keypoints[i - 1]
            next_kp = keypoints[(i + 1) % len(keypoints)]
            dist_next, angle_next = calculate_distance_angle([x, y], next_kp[:2])
            dist_prev, angle_prev = calculate_distance_angle([x, y], prev_kp[:2])
            node_features.append([x, y, dist_next, angle_next, dist_prev, angle_prev])
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
                transformed_img, pred_kp_transformed, gt_kp_transformed = self.apply_transform(img, pred_kp, gt_kp)
                # Call the visualization function
                visualize_keypoints(transformed_img, pred_kp_transformed, gt_kp_transformed, title="Transformed Keypoints")
               
                pred_kp_normalized = self.normalize_keypoints(pred_kp_transformed, transformed_img.shape[1], transformed_img.shape[0])      

                batch_transformed_data.append((pred_kp_transformed, pred_kp_normalized, gt_kp_transformed))
                

            all_graphs = [self.keypoints_to_graph(kp, transformed_img.shape[1], transformed_img.shape[0]) for _, kp, _ in batch_transformed_data]
            all_predictions = [self.graph_gcn(graph.x, graph.edge_index) for graph in all_graphs]

            return torch.stack(all_predictions), [gt_kp for _, _, gt_kp in batch_transformed_data], [init_kp for init_kp, _, _ in batch_transformed_data], transformed_img.shape[1], transformed_img.shape[0]
        
        else:  # Inference Mode
            # Step 5: Convert keypoints to graph representations
            all_graphs = [
                self.keypoints_to_graph(kp, 640, 480)  # Assume fixed image size; adapt as needed
                for kp in batch_pred_keypoints
            ]

            # Step 6: Pass graphs through Graph GCN
            all_predictions = [
                self.graph_gcn(graph.x, graph.edge_index) for graph in all_graphs
            ]

            # Step 7: Return Graph GCN predictions (refined keypoints)
            return all_predictions
 

def kgnn2d_loss(gt_keypoints, pred_keypoints, gt_distances_next, gt_angles_next, gt_distances_prev, gt_angles_prev, pred_distances_next, pred_angles_next, pred_distances_prev, pred_angles_prev):
    keypoints_loss = func.mse_loss(pred_keypoints, gt_keypoints)
    prev_distances_loss = func.mse_loss(pred_distances_prev, gt_distances_prev)
    prev_angles_loss = func.mse_loss(pred_angles_prev, gt_angles_prev)
    next_distances_loss = func.mse_loss(pred_distances_next, gt_distances_next)
    next_angles_loss = func.mse_loss(pred_angles_next, gt_angles_next)
    return keypoints_loss + prev_distances_loss + prev_angles_loss + next_distances_loss + next_angles_loss

train_model = KeypointV2Pipeline(weights_path, model_state_dict)
train_model = train_model.to(device)

optimizer = torch.optim.Adam(train_model.parameters(), lr=0.0001)
scaler = GradScaler()

num_epochs = 100
batch_size = 128

split_folder_path = train_test_split(root_dir)
KEYPOINTS_FOLDER_TRAIN = split_folder_path +"/train"
KEYPOINTS_FOLDER_VAL = split_folder_path +"/val"
KEYPOINTS_FOLDER_TEST = split_folder_path +"/test"

dataset_train = GTDataset(KEYPOINTS_FOLDER_TRAIN)
dataset_val = GTDataset(KEYPOINTS_FOLDER_VAL)
dataset_test = GTDataset(KEYPOINTS_FOLDER_TEST)

data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

checkpoint_dir = '/home/schatterjee/lama/kprcnn_panda/trained_models/gcn_sage_ckpt_v2/'
checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')

# Create checkpoint directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

# Load checkpoint if exists
start_epoch = 0
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    train_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from epoch {start_epoch}")

start_time = time.time()
for epoch in range(start_epoch, num_epochs):
    train_model.train()
    total_loss = 0

    for imgs, target_dicts, _ in data_loader_train:
        imgs = [img.to(device) for img in imgs]
        optimizer.zero_grad()
        gt_keypoints_batch = process_batch_keypoints(target_dicts)

        with autocast():
            KGNN2D, transformed_gt_keypoints, init_kp_denorm, width, height = train_model(imgs, gt_keypoints_batch)
            denormalized_keypoints = denormalize_keypoints(KGNN2D, width, height)
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
        checkpoint_path = os.path.join(checkpoint_dir, f'kprcnn_occ_gcn_sage_ckpt_v2_b{batch_size}e{epoch+1}.pth')
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
model_save_path = f"/home/schatterjee/lama/kprcnn_panda/trained_models/kprcnn_gcn_sage_v2_b{batch_size}_e{num_epochs}.pth"
torch.save(model.state_dict(), model_save_path)




