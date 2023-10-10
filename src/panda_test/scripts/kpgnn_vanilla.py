#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# to generalize home directory. User can change their parent path without entering their home directory
path = Def_Path()

parent_path =  path.home + "/Pictures/" + "Data/"

root_dir = parent_path + path.year + "-" + path.month + "-" + path.day + "/"


# In[3]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.cuda.set_per_process_memory_fraction(0.9, 0)
print(device)


# In[4]:


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


# In[5]:


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
        
    output = parent_path + "split_folder_output" + "-" + path.year + "-" + path.month + "-" + path.day 
    
    splitfolders.ratio(src_dir, # The location of dataset
                   output=output, # The output location
                   seed=42, # The number of seed
                   ratio=(.7, .2, .1), # The ratio of split dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
                   )
    
    shutil.rmtree(dst_dir_img)
    shutil.rmtree(dst_dir_anno)
    
    return output  
    


# In[6]:


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
        labels = [1, 2, 3, 4, 5, 6]            
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
            return img, target, img_original, target_original
        else:
            return img, target
    
    def __len__(self):
        return len(self.imgs_files)


# In[7]:


import torch.nn as nn

class VanillaGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VanillaGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, nodes, edges):
        # Aggregate neighboring node features
        summed_neighbor_nodes = torch.zeros_like(nodes)
        for (src, dest) in edges:
            summed_neighbor_nodes[dest] += nodes[src]

        # Pass through neural network layers
        out = torch.relu(self.fc1(summed_neighbor_nodes))
        out = self.fc2(out)
        return out


# In[8]:


class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.keypoint_rcnn = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True, num_keypoints=6, num_classes=7)
        self.gnn1 = VanillaGNN(2, 64, 2)
#         self.gnn2 = VanillaGNN(64, 64, 64)
#         self.gnn3 = VanillaGNN(64, 64, 2)
    
    def complete_missing_keypoints(self, keypoints, labels, num_expected_keypoints=6):

        detected_kps = keypoints.shape[0]
        # Check if all keypoints are detected
        if detected_kps == num_expected_keypoints:
            return keypoints

        # Create a placeholder tensor for keypoints with the correct shape
        ordered_keypoints = torch.zeros((num_expected_keypoints, 3), device=keypoints.device)

        # If some keypoints are detected, compute their average position
        average_kp = torch.mean(keypoints, dim=0, keepdim=True)

        for i, label in enumerate(labels):
            ordered_keypoints[label - 1] = keypoints[i]
            
        # Fill in the missing keypoints with the average position
        missing_indices = (torch.sum(ordered_keypoints, dim=1) == 0).nonzero(as_tuple=True)[0]
        ordered_keypoints[missing_indices] = average_kp
        
        return ordered_keypoints
    
    
    def get_edge_features(self):
        
        edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
        return edges
    

    def forward(self, images, adj_matrix=None, targets=None, train=False):
        if train:
            output = self.keypoint_rcnn(images, targets)
            return output
    
        else:
            all_keypoints = []
            with torch.no_grad():
                self.keypoint_rcnn.eval()
                output = self.keypoint_rcnn(images)
                self.keypoint_rcnn.train()
                
                keypoints = output[0]['keypoints'].detach().cpu().numpy()
                kp_score = output[0]['keypoints_scores'].detach().cpu().numpy()
                labels = output[0]['labels'].detach().cpu().numpy()
                unique_labels = list(set(labels))
                scores = output[0]['scores'].detach().cpu().numpy()
                print("labels", unique_labels)
                kps = []
                kp_scores = []
                ulabels = []

                for label in unique_labels:
                    indices = [j for j, x in enumerate(labels) if x == label]
                    scores_for_label = [scores[j] for j in indices]
                    max_score_index = indices[scores_for_label.index(max(scores_for_label))]
                    kp_score_label = kp_score[max_score_index].tolist()
                    kps.append(keypoints[max_score_index][kp_score_label.index(max(kp_score_label))])
                    ulabels.append(label)

                kps = [torch.tensor(kp, dtype=torch.float32) for kp in kps]
                if not kps:
                    default_value = torch.tensor([[320, 240, 1]], dtype=torch.float32, device=images[i].device)
                    keypoints = default_value.repeat(6, 1)
                else:
                    keypoints = torch.stack(kps)
                        
                print("kp before placeholder", keypoints)
                keypoints = self.complete_missing_keypoints(keypoints, unique_labels)[:, 0:2].to(device)
                print("kp after placeholder", keypoints)
                edges = self.get_edge_features().to(device)
                keypoints = self.gnn1(keypoints, edges)
                print("kp first layer", keypoints)
#                 keypoints = self.gnn2(keypoints, edges)
#                 print("kp second layer iteration", keypoints)
#                 keypoints = self.gnn3(keypoints, edges)
#                 keypoints = nn.functional.relu(keypoints)
                print("kp after graph", keypoints)
            print("All keypoints", keypoints)

            return keypoints


# In[9]:


# Initialize model and optimizer
model = CombinedModel()
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 100
batch_size = 4

KEYPOINTS_FOLDER_TRAIN = train_test_split(root_dir) +"/train" #train_test_split(root_dir) +"/train"
KEYPOINTS_FOLDER_VAL = train_test_split(root_dir) +"/val"
KEYPOINTS_FOLDER_TEST = train_test_split(root_dir) +"/test"

dataset_train = KPDataset(KEYPOINTS_FOLDER_TRAIN, transform=train_transform(), demo=False)
dataset_val = KPDataset(KEYPOINTS_FOLDER_VAL, transform=None, demo=False)
dataset_test = KPDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)

data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Initialize the GradScaler for mixed precision training
scaler = GradScaler()

top_5_models = []

model.train()
mse_loss = nn.MSELoss()

for epoch in range(num_epochs):  # for 50 epochs
    for batch_idx, batch in enumerate(data_loader_train):
        images, targets = batch          
        # Move images to GPU
        images = torch.stack(images).cuda()
        imgs = [img.to(device) for img in images]  # Create list of images
        # Move targets to GPU
        for target in targets:
            for key, val in target.items():
                target[key] = val.cuda()
        
        optimizer.zero_grad()
        individual_losses = []
        
       # Automatic mixed precision for forward pass
        with autocast():
            output_train = model(images, targets=targets, train=True)
    
            for i in range(len(images)):
                img = images[i].unsqueeze(0).to(device)
                ground_truth_keypoints = targets[i]['keypoints'].to(device).squeeze()[:,0:2]
                print("ground truth keypoints shape", ground_truth_keypoints)
                optimizer.zero_grad()
                # automatic precision for forward pass
                predicted_keypoints = model(img, train=False)    
                # Compute the loss for this image
                # Compute loss
                loss = mse_loss(predicted_keypoints, ground_truth_keypoints)
                individual_losses.append(loss.item())
            
            # Aggregate the individual losses to get a scalar loss
            scalar_loss = sum(individual_losses) / len(individual_losses)      
            loss_keypoint = output_train['loss_keypoint']
            
            total_loss = scalar_loss + 0.01*loss_keypoint  
        print("total_loss", total_loss)
        
        # Scale the loss and backpropagate
        scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scheduler.step()
        scaler.step(optimizer)
        scaler.update()
        
        # Check if the current model should be saved as a top model
        if len(top_5_models) < 5 or loss.item() < max(top_5_models, key=lambda x: x[0])[0]:
            # Save the model state and loss
            model_state = {
                'epoch': epoch,
                'complete_model': model,
#                 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
            }
            top_5_models.append((total_loss.item(), model_state))

            # Sort the list based on loss (ascending order)
            top_5_models.sort(key=lambda x: x[0])

            # If there are more than 5 models, remove the one with the highest loss
            if len(top_5_models) > 5:
                top_5_models.pop()

        print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx + 1}/{len(data_loader_train)}, Loss: {total_loss.item()}")
        
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item()}")
    
# After all epochs, save the top 5 models to disk
for idx, (_, model_state) in enumerate(top_5_models):
    torch.save(model_state, f'/home/jc-merlab/Pictures/Data/trained_models/vanilla_gnn_model_b{batch_size}_e{num_epochs}_{idx+1}.pth')


