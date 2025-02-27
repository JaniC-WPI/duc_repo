import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

weights_path = '/home/schatterjee/lama/kprcnn_panda/trained_models/keypointsrcnn_planning_b1_e50_v8.pth'

n_nodes = 9

# to generalize home directory. User can change their parent path without entering their home directory
path = Def_Path()

# parent_path =  path.home + "/Pictures/" + "Data/"

parent_path =  "/home/schatterjee/lama/kprcnn_panda/"

# parent_path = "/media/jc-merlab/Crucial X9/"

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
        
    output = parent_path + "split_folder_output_gcn_sage" + "-" + path.year + "-" + path.month + "-" + path.day 
    
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


class GCN_SAGE_HYBRID(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_1, hidden_channels_2, out_channels, parallel=False, dropout_prob=0.5):
        """
        Graph Neural Network that supports both Sequential (SAGE → GCN) and Parallel (SAGE + GCN).
        
        Args:
            in_channels (int): Number of input features per node.
            hidden_channels_1 (int): Hidden size for the first set of layers.
            hidden_channels_2 (int): Hidden size for the second set of layers.
            out_channels (int): Output feature size.
            parallel (bool): If True, uses Parallel Network (GraphSAGE + GCN). If False, uses Sequential (GraphSAGE → GCN).
            dropout_prob (float): Dropout probability (default = 0.5).
        """
        super(GCN_SAGE_HYBRID, self).__init__()
        self.parallel = parallel
        self.dropout_prob = dropout_prob

        if parallel:
            # **Parallel Model (GraphSAGE + GCN)**
            self.sage1 = SAGEConv(in_channels, hidden_channels_1)
            self.gcn1 = GCNConv(in_channels, hidden_channels_1)
            
            self.sage2 = SAGEConv(hidden_channels_1, hidden_channels_2)
            self.gcn2 = GCNConv(hidden_channels_1, hidden_channels_2)
            
            self.fc = nn.Linear(hidden_channels_2 * 2, out_channels)  # Concatenated output size = 2x hidden_channels_2
        
        else:
            # **Sequential Model (GraphSAGE → GCN)**
            self.sage1 = SAGEConv(in_channels, hidden_channels_1)
            self.sage2 = SAGEConv(hidden_channels_1, hidden_channels_1)
            
            self.gcn1 = GCNConv(hidden_channels_1, hidden_channels_2)
            self.gcn2 = GCNConv(hidden_channels_2, hidden_channels_2)
            
            self.fc = nn.Linear(hidden_channels_2, out_channels)

    def forward(self, x, edge_index):
        x = x.cuda()
        edge_index = edge_index.cuda()

        if self.parallel:
            # **Parallel Forward Pass (GraphSAGE + GCN)**
            x_sage = func.relu(self.sage1(x, edge_index))
            x_sage = func.dropout(x_sage, p=self.dropout_prob, training=self.training)

            x_sage = func.relu(self.sage2(x_sage, edge_index))
            x_sage = func.dropout(x_sage, p=self.dropout_prob, training=self.training)

            x_gcn = func.relu(self.gcn1(x, edge_index))
            x_gcn = func.dropout(x_gcn, p=self.dropout_prob, training=self.training)

            x_gcn = func.relu(self.gcn2(x_gcn, edge_index))
            x_gcn = func.dropout(x_gcn, p=self.dropout_prob, training=self.training)

            # Concatenating outputs from both branches
            x = torch.cat([x_sage, x_gcn], dim=1)

        else:
            # **Sequential Forward Pass (GraphSAGE → GCN)**
            x = func.relu(self.sage1(x, edge_index))
            x = func.dropout(x, p=self.dropout_prob, training=self.training)

            x = func.relu(self.sage2(x, edge_index))
            x = func.dropout(x, p=self.dropout_prob, training=self.training)

            x = func.relu(self.gcn1(x, edge_index))
            x = func.dropout(x, p=self.dropout_prob, training=self.training)

            x = func.relu(self.gcn2(x, edge_index))
            x = func.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.fc(x)
        return x

class KeypointPipeline(nn.Module):
    def __init__(self, weights_path, hidden_channels_1=1024, hidden_channels_2=512, parallel=False, dropout_prob=0.5):
        super(KeypointPipeline, self).__init__()
        self.keypoint_model = torch.load(weights_path).to(device)
        self.gcn_sage = GCN_SAGE_HYBRID(8, hidden_channels_1, hidden_channels_2, 4, parallel, dropout_prob).to(device)

        # 🔹 Apply Image Quality Transformations Before Keypoint R-CNN
        self.image_quality_transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=0.5)
        ])

        # 🔹 Apply Geometric Transformations After Keypoint R-CNN
        self.geometric_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.OneOf([
                A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.2), rotate=(-10, 10), shear=(-5, 5)),
                A.NoOp()
            ], p=0.5),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

    def process_model_output(self, output):
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.01)[0].tolist()
        confidence = output[0]['scores'][high_scores_idxs].detach().cpu().numpy()
        labels = output[0]['labels'][high_scores_idxs].detach().cpu().numpy()
        keypoints = [list(map(int, kps[0, 0:2])) + [confidence[idx]] + [labels[idx]]
                     for idx, kps in enumerate(output[0]['keypoints'][high_scores_idxs].detach().cpu().numpy())]

        keypoints = torch.stack([torch.tensor(kp, dtype=torch.float32).to(device) for kp in keypoints]).to(device)
        return keypoints

    def apply_quality_transform(self, img):
        img_np = img.permute(1, 2, 0).cpu().numpy()
        transformed = self.image_quality_transform(image=img_np)
        return F.to_tensor(transformed['image']).to(device)
    
    def apply_geometric_transform(self, img, pred_keypoints, gt_keypoints=None):
        img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert (C, H, W) -> (H, W, C)
        
        # Prepare keypoints for transformation with all attributes (x, y, confidence, label)
        pred_keypoints_np = [kp[:2].tolist() for kp in pred_keypoints]
        pred_attributes = [kp[2:].tolist() for kp in pred_keypoints]

        pred_keypoints_np = pred_keypoints[:, :2].cpu().numpy().tolist()

        gt_keypoints_np = [kp[:2].tolist() for kp in gt_keypoints] if gt_keypoints is not None else []

        # Get the original image height and width
        orig_height, orig_width = img_np.shape[:2]
        
        if gt_keypoints_np:
            gt_x = [kp[0] for kp in gt_keypoints_np]
            gt_y = [kp[1] for kp in gt_keypoints_np]
            min_x, max_x = min(gt_x), max(gt_x)
            min_y, max_y = min(gt_y), max(gt_y)
        else:
            min_x, max_x, min_y, max_y = 0, orig_width, 0, orig_height  # No GT keypoints, use full image

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
        
        post_prediction_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.OneOf([
                A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-10, 10), shear=(-5, 5)),
                A.NoOp()
            ], p=0.5),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


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
                                  [[(i + 1) % len(keypoints), i] for i in range(len(keypoints))],
                                  dtype=torch.long).t().contiguous().to(device)
        return Data(x=node_features, edge_index=edge_index)

    def forward(self, imgs, gt_keypoints_batch=None):
        keypoint_model_training = self.keypoint_model.training
        self.keypoint_model.eval()

        # Step 1: Apply Image Quality Transformations Before Keypoint R-CNN
        transformed_imgs = []
        for img in imgs:
            # 🔹 Apply Quality Transformations Before Keypoint R-CNN
            img_np = img.permute(1, 2, 0).cpu().numpy()
            transformed = self.image_quality_transform(image=img_np)
            transformed_img = F.to_tensor(transformed['image']).to(device)
            transformed_imgs.append(transformed_img)


        with torch.no_grad():
            batch_outputs = [self.keypoint_model(img.unsqueeze(0).to(device)) for img in transformed_imgs]

        self.keypoint_model.train(mode=keypoint_model_training)

        batch_pred_keypoints = [self.process_model_output(output) for output in batch_outputs]

        if gt_keypoints_batch is not None:  # Training Mode
            batch_transformed_data = []
            for img, pred_kp, gt_kp in zip(transformed_imgs, batch_pred_keypoints, gt_keypoints_batch):
                transformed_img, pred_kp_transformed, gt_kp_transformed = self.apply_geometric_transform(img, pred_kp, gt_kp)

                pred_kp_filled = self.fill_missing_keypoints(pred_kp_transformed, transformed_img.shape[1], transformed_img.shape[0])
                pred_kp_normalized = self.normalize_keypoints(pred_kp_filled, transformed_img.shape[1], transformed_img.shape[0])

                batch_transformed_data.append((pred_kp_filled[:, :2], pred_kp_normalized, gt_kp_transformed))

            all_graphs = [self.keypoints_to_graph(kp, transformed_img.shape[1], transformed_img.shape[0]) for _, kp, _ in batch_transformed_data]
            all_predictions = [self.gcn_sage(graph.x, graph.edge_index) for graph in all_graphs]

            return torch.stack(all_predictions), [gt_kp for _, _, gt_kp in batch_transformed_data], [init_kp for init_kp, _, _ in batch_transformed_data], transformed_img.shape[1], transformed_img.shape[0]

        else:  # Inference Mode
            batch_final_keypoints = []
            for pred_kp in batch_pred_keypoints:
                pred_kp_filled = self.fill_missing_keypoints(pred_kp, 640, 480)
                pred_kp_normalized = self.normalize_keypoints(pred_kp_filled, 640, 480)
                batch_final_keypoints.append(pred_kp_normalized)

            return batch_final_keypoints  


def kprcnn_loss(gt_keypoints, pred_keypoints, visibility, occlusion_weight=2.0):
    base_loss = func.mse_loss(pred_keypoints, gt_keypoints, reduction='none')
#     visibility.unsqueeze(-1).expand(-1, -1, 2)
    
    weighted_loss = base_loss * (1 + (occlusion_weight - 1) * (1 - visibility))
    return weighted_loss.mean()


def graph_loss(gt_keypoints, pred_keypoints, gt_distances_next, gt_angles_next, 
                gt_distances_prev, gt_angles_prev, pred_distances_next, pred_angles_next, 
                pred_distances_prev, pred_angles_prev, visibility, occlusion_weight=5.0):
    """
    Compute loss with weighted penalty for occluded keypoints.

    Args:
        gt_keypoints (Tensor): Ground truth keypoints (B, 9, 2)
        pred_keypoints (Tensor): Predicted keypoints (B, 9, 2)
        gt_distances_next (Tensor): Ground truth distances to next keypoint
        gt_angles_next (Tensor): Ground truth angles to next keypoint
        gt_distances_prev (Tensor): Ground truth distances to previous keypoint
        gt_angles_prev (Tensor): Ground truth angles to previous keypoint
        pred_distances_next (Tensor): Predicted distances to next keypoint
        pred_angles_next (Tensor): Predicted angles to next keypoint
        pred_distances_prev (Tensor): Predicted distances to previous keypoint
        pred_angles_prev (Tensor): Predicted angles to previous keypoint
        visibility (Tensor): Visibility values (B, 9) where 0 = occluded, 1 = visible
        occlusion_weight (float): Weight factor for occluded keypoints (default = 5.0)

    Returns:
        loss (Tensor): Weighted loss value
    """
    # Compute base losses
    keypoints_loss = func.mse_loss(pred_keypoints, gt_keypoints, reduction='none')  # (B, 9, 2)
    prev_distances_loss = func.mse_loss(pred_distances_prev, gt_distances_prev, reduction='none')  # (B, 9)
    prev_angles_loss = func.mse_loss(pred_angles_prev, gt_angles_prev, reduction='none')  # (B, 9)
    next_distances_loss = func.mse_loss(pred_distances_next, gt_distances_next, reduction='none')  # (B, 9)
    next_angles_loss = func.mse_loss(pred_angles_next, gt_angles_next, reduction='none')  # (B, 9)

    # Expand visibility to match shape of keypoints
#     visibility = visibility.unsqueeze(-1)  # (B, 9, 1) for keypoints

    # Compute occlusion weights
    weights = torch.where(visibility == 0, occlusion_weight, 1.0)  # (B, 9, 1)
    weights_dist = torch.where(visibility.squeeze(-1) == 0, occlusion_weight, 1.0)  # (B, 9) for distance/angle losses

    # Apply weights to the losses
    weighted_keypoints_loss = (keypoints_loss * weights).mean()
    weighted_prev_distances_loss = (prev_distances_loss * weights_dist).mean()
    weighted_prev_angles_loss = (prev_angles_loss * weights_dist).mean()
    weighted_next_distances_loss = (next_distances_loss * weights_dist).mean()
    weighted_next_angles_loss = (next_angles_loss * weights_dist).mean()

    # Total loss
    loss = (weighted_keypoints_loss + 
            weighted_prev_distances_loss + weighted_prev_angles_loss +
            weighted_next_distances_loss + weighted_next_angles_loss)

    return loss

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


## Optune tuning

import optuna

import random

class KPDatasetSubset(KPDataset):
    def __init__(self, root, transform=None, demo=False, fraction=0.2):
        super().__init__(root, transform, demo)
        
        # Compute subset size
        subset_size = int(len(self.imgs_files) * fraction)

        # Randomly shuffle indices and select subset
        indices = list(range(len(self.imgs_files)))
        random.shuffle(indices)
        selected_indices = indices[:subset_size]

        # Filter images and annotations based on selected indices
        self.imgs_files = [self.imgs_files[i] for i in selected_indices]
        self.annotations_files = [self.annotations_files[i] for i in selected_indices]

split_folder_path = train_test_split(root_dir)
KEYPOINTS_FOLDER_TRAIN = split_folder_path +"/train"
KEYPOINTS_FOLDER_VAL = split_folder_path +"/val"
KEYPOINTS_FOLDER_TEST = split_folder_path +"/test"


# Training objective function with Optuna
def objective(rank, world_size, trial):
    setup(rank, world_size)
	
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    dropout_prob = trial.suggest_uniform("dropout_prob", 0.2, 0.6)
    hidden_channels_1 = trial.suggest_int("hidden_channels_1", 512, 2048, step=256)
    hidden_channels_2 = trial.suggest_int("hidden_channels_2", 256, 1024, step=128)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    parallel = trial.suggest_categorical("parallel", [True, False])
    num_epochs = trial.suggest_int("num_epochs", 5, 20, 50, 100, step=5)
    
    if optimizer_name == "SGD":
        momentum = trial.suggest_uniform("momentum", 0.5, 0.99)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler_name = trial.suggest_categorical("scheduler", ["StepLR", "ExponentialLR", "ReduceLROnPlateau", "CosineAnnealingLR"])

    if scheduler_name == "StepLR":
        step_size = trial.suggest_int("step_size", 10, 50, step=5)  # Decay LR every few epochs
        gamma = trial.suggest_uniform("gamma", 0.1, 0.5)  # Factor to reduce LR
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_name == "ExponentialLR":
        gamma = trial.suggest_uniform("gamma_exp", 0.85, 0.99)  # Exponential decay rate
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    elif scheduler_name == "ReduceLROnPlateau":
        factor = trial.suggest_uniform("factor", 0.1, 0.5)  # LR reduction factor
        patience = trial.suggest_int("patience", 3, 10)  # Number of epochs before reducing LR
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)

    else:  # CosineAnnealingLR
        T_max = trial.suggest_int("T_max", 10, 50, step=5)  # Number of epochs for full cycle
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    device = torch.device(f"cuda:{rank}")
    
    model = KeypointPipeline(weights_path, hidden_channels_1=hidden_channels_1, hidden_channels_2=hidden_channels_2, parallel=parallel, dropout_prob=dropout_prob)
    model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler()

    dataset_train = KPDatasetSubset(KEYPOINTS_FOLDER_TRAIN, transform=None, fraction=0.2)
    train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank)
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler, num_workers=4)

    for epoch in range(num_epochs):
        model.train()
        if train_sampler:
    	    train_sampler.set_epoch(epoch)
        total_loss = 0

        for imgs, keypoints in data_loader_train:
            imgs, keypoints = imgs.to(device), keypoints.to(device)
            optimizer.zero_grad()

            with autocast():
                output = model(keypoints, torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(device))
                loss = func.mse_loss(output, keypoints)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader_train)
        trial.report(avg_loss, epoch)
        
        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(avg_loss)  # Use loss for ReduceLROnPlateau
        else:
            scheduler.step()  # Normal schedulers

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_loss

if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    rank = int(os.environ.get("RANK", 0))
    setup(rank, world_size)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), pruner=pruner, direction="minimize")
    study.optimize(objective, n_trials=150)
    best_loss = objective(rank, world_size, study)
    
    print(f"Rank {rank} - Best Loss: {best_loss}")
    print(f"Rank {rank} - Best Hyperparameters: {study.best_params}")

    cleanup()
    
    # 🔹 Get best trial
    best_params = study.best_params

    # 🔹 Save best hyperparameters to a JSON file
    log_file = "/home/schatterjee/lama/kprcnn_panda/panda_test/best_hyperparams.json"
    with open(log_file, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Best hyperparameters saved to {log_file}")
    print("Best hyperparameters:", best_params)
