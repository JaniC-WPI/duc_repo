#!/usr/bin/env python
# coding: utf-8

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
from torch_geometric.nn import SAGEConv, GCNConv
import torch.nn.functional as func
from torch_geometric.data import Data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchsummary import summary
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split

import albumentations as A # Library for augmentations

import matplotlib.pyplot as plt 
from PIL import Image

import transforms, utils, engine, train
from utils import collate_fn
from engine import train_one_epoch, evaluate


t = torch.cuda.get_device_properties(0).total_memory
print(t)
torch.cuda.empty_cache()

r = torch.cuda.memory_reserved(0)
print(r)
a = torch.cuda.memory_allocated(0)
print(a)
# f = r-a  # free inside reserved

weights_path = '/home/schatterjee/lama/kprcnn_panda/trained_models/keypointsrcnn_weights_sim_b1_e25_v0.pth'

# to generalize home directory. User can change their parent path without entering their home directory
path = Def_Path()

parent_path =  "/home/schatterjee/lama/kprcnn_panda/"

# root_dir = parent_path + path.year + "-" + path.month + "-" + path.day + "/"
root_dir = parent_path + "occ_sim_dataset/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.cuda.set_per_process_memory_fraction(0.9, 0)
print(device)

# this fucntion tranforms an input image for diverseifying data for training
def train_transform():
    return A.Compose([
        A.Sequential([
            A.RandomRotate90(p=1), 
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=1), 
        ], p=1),
        A.Resize(640, 480),  # Resize every image to 640x480 after all other transformations
    ],
    keypoint_params=A.KeypointParams(format='xy'),
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels'])
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
        
    output = parent_path + "split_folder_occ_sage" + "-" + path.year + "-" + path.month + "-" + path.day 
    
    splitfolders.ratio(src_dir, # The location of dataset
                   output=output, # The output location
                   seed=42, # The number of seed
                   ratio=(.8, .1, .1), # The ratio of split dataset
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

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
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
        x = self.conv3(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.fc(x)
        return x

class KeypointPipeline(nn.Module):
    def __init__(self, weights_path):
        super(KeypointPipeline, self).__init__()  
        self.keypoint_model = torch.load(weights_path).to(device)
        self.graph_sage = GraphSAGE(4,1024,4)          
    
    def process_keypoints_and_edges(self, output, num_expected_keypoints=6):
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0]

        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], 
                                            output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()

        confidence = output[0]['scores'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()
        labels = output[0]['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()
        keypoints_list = []
        for idx, kps in enumerate(output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()):
            keypoints_list.append(list(map(int, kps[0, :2])) + [confidence[idx]] + [labels[idx]])

        # Initialize an array to hold processed keypoints with default confidence score -1 and label 0
        processed_kps = np.full((num_expected_keypoints, 4), -1.0, dtype=np.float32)

        # Populate the processed keypoints with the detected ones
        for kp in keypoints_list:
            label = int(kp[3]) - 1
            processed_kps[label, :2] = kp[:2]
            processed_kps[label, 2] = kp[2]
            processed_kps[label, 3] = kp[3]  # Set the label

        # Handle missing keypoints
        for i, kp in enumerate(processed_kps):
            if kp[0] == -1:
                # Set label to 0 for missing keypoints
                processed_kps[i, 3] = 0

                # Compute the average of adjacent keypoints
                if i > 0 and i < num_expected_keypoints - 1 and processed_kps[i-1, 0] != -1 and processed_kps[i+1, 0] != -1:
                    processed_kps[i, :2] = (processed_kps[i-1, :2] + processed_kps[i+1, :2]) / 2
                elif i == 0 and processed_kps[1, 0] != -1:
                    processed_kps[i, :2] = processed_kps[1, :2]  # Use next keypoint if first one is missing
                elif i == num_expected_keypoints - 1 and processed_kps[i-1, 0] != -1:
                    processed_kps[i, :2] = processed_kps[i-1, :2]  # Use previous keypoint if last one is missing
                else:
                    processed_kps[i, :2] = [320, 240]  # Center of the image (assuming 640x480 resolution)

                processed_kps[i, 2] = 0  # Set confidence to 0

        # Convert the numpy array back to a torch tensor
        processed_kps_tensor = torch.from_numpy(processed_kps).to(device)

        return processed_kps_tensor
    
    def get_edges(self):
        edges = [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]]
        return torch.tensor(edges, dtype=torch.long)
    
    def forward(self, imgs):
        # Temporarily set the keypoint model to evaluation mode
        keypoint_model_training = self.keypoint_model.training
        self.keypoint_model.eval()

        # Process each image in the batch
        with torch.no_grad():
            batch_outputs = [self.keypoint_model(img.unsqueeze(0).to(device)) for img in imgs]

        # Set the keypoint model back to its original training mode
        self.keypoint_model.train(mode=keypoint_model_training)

        # Process model outputs using process_keypoints_and_edges
        batch_labeled_keypoints = [self.process_keypoints_and_edges(output) for output in batch_outputs]

        # Generate graph input tensor for each set of labeled keypoints
        batch_x = [labeled_keypoints for labeled_keypoints in batch_labeled_keypoints]

        # Stack the batch of x tensors for batch processing
        batch_x = torch.stack(batch_x)

        # Get the edges for the graph representation
        edges = self.get_edges()

        # Perform a forward pass of the GraphSAGE model on each graph in the batch
        batch_keypoints = [self.graph_sage(x_i, edges) for x_i in batch_x]

        # Return the keypoints for each image in the batch
        return torch.stack(batch_keypoints), batch_labeled_keypoints

def kgnn2d_loss(gt_keypoints, pred_keypoints, visibility):
    # Assuming visibility is 0 for occluded and 1 for visible keypoints
    visibility=visibility.unsqueeze(1)
    weights = torch.ones_like(visibility)
    weights[visibility == 0] = 2  # Increase weight for occluded keypoints
#     weights.unsqueeze(-1)
    print("Weights", weights)
    loss = func.mse_loss(pred_keypoints * weights, gt_keypoints * weights)
    return loss

def process_keypoints(keypoints):
    # Assuming keypoints is a list of Nx3 tensors where N is the number of keypoints
    # and each keypoint is represented as [x, y, visibility]
    # Remove the unnecessary middle dimension
    keypoints = [kp.squeeze(1) for kp in keypoints]
    visibilities = [kp[:, 2] for kp in keypoints]  # Extract visibility flags
    valid_vis_all = torch.cat([v == 1 for v in visibilities]).long().cuda()
    valid_invis_all = torch.cat([v == 0 for v in visibilities]).long().cuda()

    keypoints_gt = torch.cat([kp[:, :2] for kp in keypoints]).float().cuda()  # Gather all keypoints and discard visibility flags
    keypoints_gt = keypoints_gt.view(-1, 2).unsqueeze(0)  # Add an extra dimension to match expected shape for loss_edges

    return keypoints_gt, valid_vis_all, valid_invis_all


# Initialize model and optimizer
model = KeypointPipeline(weights_path)
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 200
batch_size = 64

split_folder_path = train_test_split(root_dir)

KEYPOINTS_FOLDER_TRAIN = split_folder_path +"/train" #train_test_split(root_dir) +"/train"
KEYPOINTS_FOLDER_VAL = split_folder_path +"/val"
KEYPOINTS_FOLDER_TEST = split_folder_path +"/test"

dataset_train = KPDataset(KEYPOINTS_FOLDER_TRAIN, transform=None, demo=False)
dataset_val = KPDataset(KEYPOINTS_FOLDER_VAL, transform=None, demo=False)
dataset_test = KPDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)

data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

v = 1

total_start_time = datetime.now()
loss_history = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch in data_loader_train:
        imgs, target_dicts, _ = batch
#         imgs = torch.stack(imgs)
        imgs = [img.to(device) for img in imgs]
        optimizer.zero_grad()
        # Forward pass
        pred_keypoints, init_keypoints = model([img.to(device) for img in imgs])

        # Prepare batch loss computation
        batch_losses = []
        for i in range(len(imgs)):
            # Process each image in the batch
            keypoints = [target_dicts[i]['keypoints']]
            print("gt keypoints", keypoints)
            keypoints_gt, valid_vis_all, valid_invis_all = process_keypoints([target_dicts[i]['keypoints'].to(device)])
            print("init_keypoints", init_keypoints[i])
            print("pred_keypoints", pred_keypoints[i])
            loss_kgnn2d = kgnn2d_loss(keypoints_gt.squeeze(0), pred_keypoints[i][:,0:2], valid_vis_all)
#             loss = edge_loss + loss_kgnn2d
            print("loss", loss_kgnn2d)
            batch_losses.append(loss_kgnn2d)
    # Average loss over the batch and backpropagation
        batch_loss = torch.mean(torch.stack(batch_losses))
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
    
    epoch_avg_loss = epoch_loss / len(data_loader_train)
    loss_history.append(epoch_avg_loss)
    # Update the scheduler
    scheduler.step()
    
    # Calculate time elapsed for the epoch
    epoch_end_time = datetime.now()
    epoch_time = (epoch_end_time - epoch_start_time).total_seconds()
    total_time_elapsed = (epoch_end_time - total_start_time).total_seconds()
    average_time_per_epoch = total_time_elapsed / (epoch + 1)
    eta = average_time_per_epoch * (num_epochs - epoch - 1)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(data_loader_train)}, Epoch Time: {epoch_time}s, ETA: {eta:.2f}s')
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss_history': loss_history
        }
        checkpoint_path = f"/home/schatterjee/lama/kprcnn_panda/trained_models/checkpoints/occ_sage_epoch_{epoch + 1}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

# # Save the model after training
# torch.save(model.state_dict(), 'model.pth')

model_save_path = f"/home/schatterjee/lama/kprcnn_panda/trained_models/krcnn_occ_sage_b{batch_size}_e{num_epochs}_v{v}.pth"

torch.save(model, model_save_path)





