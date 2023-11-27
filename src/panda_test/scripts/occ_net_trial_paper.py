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

# weights_path = '/home/jc-merlab/Pictures/Data/trained_models/keypointsrcnn_weights_sim_b1_e25_v0.pth'
weights_path = '/home/schatterjee/lama/kprcnn_panda/trained_models/keypointsrcnn_weights_sim_b1_e25_v0.pth'

# to generalize home directory. User can change their parent path without entering their home directory
path = Def_Path()

parent_path =  "/home/schatterjee/lama/kprcnn_panda/"

root_dir = parent_path + "occ_sim_dataset/"

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
        
    output = parent_path + "split_folder_sim_occ" + "-" + path.year + "-" + path.month + "-" + path.day 
    
    splitfolders.ratio(src_dir, # The location of dataset
                   output=output, # The output location
                   seed=42, # The number of seed
                   ratio=(.95, 0.025, 0.025), # The ratio of split dataset
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
            return img, target, img_original, target_original, img_file
        else:
            return img, target, img_file
    
    def __len__(self):
        return len(self.imgs_files)


import torch
import torch.nn as nn
import torch.nn.functional as func
import math
import numpy as np
from torch.autograd import Variable
import torch_geometric.nn as pyg
from torch_geometric.data import Data

_EPS = 1e-10

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
#         self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

#     def forward(self, inputs):
#         print(inputs)
#         # Input shape: [num_sims, num_things, num_features]
#         x = func.elu(self.fc1(inputs))
#         x = func.dropout(x, self.dropout_prob, training=self.training)
#         x = func.elu(self.fc2(x))
#         return self.batch_norm(x)

    def forward(self, inputs):
#         print("Input shape before any operations: ", inputs.shape)

        # Flatten the last two dimensions for the linear layer input
#         x = inputs.view(inputs.size(0), -1)
#         print("Input shape after flattening: ", x.shape)

        x = func.elu(self.fc1(inputs))
        x = func.dropout(x, self.dropout_prob, training=self.training)
        x = func.elu(self.fc2(x))

        # Assuming you want to maintain the second dimension for some reason
        # (like temporal sequence in a RNN), you would reshape the output
        # back to the desired shape. If not, this step is unnecessary.
        # output = x.view(inputs.size(0), inputs.size(1), -1)
        # print("Output shape after forward pass: ", output.shape)

        return x


class GraphEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(GraphEncoder, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP graph encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
#         print("x shape:", x.shape)
#         print("rel_rec shape:", rel_rec.shape)
#         print("rel_send shape:", rel_send.shape)

        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x        
        
        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)    
    
class GraphDecoder(nn.Module):

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        super(GraphDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned graph decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = func.relu(self.msg_fc1[i](pre_msg))
            msg = func.dropout(msg, p=self.dropout_prob)
            msg = func.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = func.dropout(func.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = func.dropout(func.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)
#        print(pred.shape,single_timestep_inputs.shape)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.


        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, :, :]
        #asa
        curr_rel_type = rel_type[:, :, :]
        preds=[]
        #print(curr_rel_type.shape)
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        #for step in range(0, pred_steps):
        last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                 curr_rel_type)
        preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1),
                 preds[0].size(2)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, :, :] = preds[i]

        pred_all = output[:, :, :]

        # NOTE: We potentially over-predicted (stored in future_pred). Unused.
        # future_pred = output[:, (inputs.size(1) - 1):, :, :]

        return pred_all#.transpose(1, 2).contiguous()    

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = func.softmax(trans_input,dim=0)
    return soft_max_1d.transpose(axis, 0)


class KeypointPipeline(nn.Module):
    def __init__(self, weights_path):
        super(KeypointPipeline, self).__init__()  
        self.keypoint_model = torch.load(weights_path).to(device)
        self.encoder = GraphEncoder(3,128,2,0.5,False)
        self.decoder = GraphDecoder(n_in_node=3,
                                 edge_types=2,
                                 msg_hid=128,
                                 msg_out=128,
                                 n_hid=128,
                                 do_prob=0.5,
                                 skip_first=False)
        
        # Define a unidirectional cyclical graph
        num_nodes = 6
        self.off_diag = np.zeros([num_nodes, num_nodes])
        
        # Creating a cycle: 1->2, 2->3, ..., 6->1
        for i in range(num_nodes):
            self.off_diag[i, (i + 1) % num_nodes] = 1

        # Update rel_rec and rel_send based on the new off_diag
        self.rel_rec = np.array(encode_onehot(np.where(self.off_diag)[1]), dtype=np.float32)
        self.rel_send = np.array(encode_onehot(np.where(self.off_diag)[0]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(self.rel_rec).to(device)
        self.rel_send = torch.FloatTensor(self.rel_send).to(device)


        self.encoder= self.encoder.cuda()
        self.decoder = self.decoder.cuda()
        self.rel_rec = self.rel_rec.cuda()
        self.rel_send = self.rel_send.cuda()
    
    def process_model_output(self, output):
        scores = output[0]['scores'].detach().cpu().numpy()
        high_scores_idxs = np.where(scores > 0.7)[0].tolist()

        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], 
                                            output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy()

        confidence = output[0]['scores'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()
        labels = output[0]['labels'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()
        keypoints = []
        for idx, kps in enumerate(output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy()):
            keypoints.append(list(map(int, kps[0,0:2])) + [confidence[idx]] + [labels[idx]])
        
        # Sort keypoints based on label
        keypoints.sort(key=lambda x: x[-1])
        return keypoints
    
    def keypoints_to_graph(self, keypoints, image_width, image_height):
        # keypoints is expected to be a tensor with shape (num_keypoints, 4),
        # where each keypoint is (x, y, score, label).
        # Convert all elements in keypoints to tensors if they are not already
        keypoints = [torch.tensor(kp, dtype=torch.float32).to(device) if not isinstance(kp, torch.Tensor) else kp for kp in keypoints]

        # Then stack them
        keypoints = torch.stack(keypoints).to(device)
        
        # Remove duplicates: Only keep the keypoint with the highest score for each label
        unique_labels, best_keypoint_indices = torch.unique(keypoints[:, 3], return_inverse=True)
        best_scores, best_indices = torch.max(keypoints[:, 2].unsqueeze(0) * (best_keypoint_indices == torch.arange(len(unique_labels)).unsqueeze(1).cuda()), dim=1)
        keypoints = keypoints[best_indices]       
        
        print("init keypoints in graph features", keypoints)

        # Normalize x and y to be in the range [-1, 1]
        keypoints[:, 0] = (keypoints[:, 0] - image_width / 2) / (image_width / 2)
        keypoints[:, 1] = (keypoints[:, 1] - image_height / 2) / (image_height / 2)

        # Use only x, y, and score for the graph features
        graph_features = keypoints[:, :3]  # Now shape is (num_keypoints, 3)

        # Ensure the shape is [num_keypoints, 3] before returning
        graph_features = graph_features.view(-1, 3)  # Reshape to ensure it's [num_keypoints, 3]

        return graph_features
        
    def forward(self, img):
        img = img.unsqueeze(0).to(device)
        # Temporarily set the keypoint model to evaluation mode
        keypoint_model_training = self.keypoint_model.training  # Save the current mode
        self.keypoint_model.eval()
        with torch.no_grad():
            output = self.keypoint_model(img)
        # Set the keypoint model back to its previous mode
        self.keypoint_model.train(keypoint_model_training)
        img = (img[0].permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)
        labeled_keypoints = self.process_model_output(output)
        # Initialize x with zeros for 6 nodes with 3 features each
        x = torch.zeros(1, 6, 3, device=device)

        # Generate graph input tensor
        keypoints = self.keypoints_to_graph(labeled_keypoints, 640, 480)

        # Ensure that keypoints are on the correct device and fill in x
        num_keypoints_detected = keypoints.size(0)
        if num_keypoints_detected <= 6:
            x[:, :num_keypoints_detected, :] = keypoints.to(device)
        else:
            raise ValueError("Number of keypoints detected exceeds the maximum of 6.")
        logits = self.encoder(x,self.rel_rec,self.rel_send)
        edges = my_softmax(logits, -1).to(device)
        KGNN2D = self.decoder(x, edges, self.rel_rec, self.rel_send,0)# args.prediction_steps)
        
        return logits,KGNN2D,labeled_keypoints

def loss_edges(valid_points, edges):
    off_diag = np.ones([6, 6]) - np.eye(6)
    idx =  torch.LongTensor(np.where(off_diag)[1].reshape(6,5)).cuda()
    if valid_points.ndim == 1:
        valid_points = valid_points.unsqueeze(0)  # Reshape to 2D if necessary

    relations = torch.zeros(valid_points.shape[0],valid_points.shape[1]*(valid_points.shape[1]-1)).cuda()
    for count,vis in enumerate(valid_points):
        vis = vis.view(-1,1) 
        vis = vis*vis.t()
        vis = torch.gather(vis,1,idx)
        relations[count] = vis.view(-1)
    relations = relations.type(torch.LongTensor).cuda() 
    loss_edges = func.cross_entropy(edges.view(-1, 2), relations.view(-1))
    return loss_edges

   
def kgnn2d_loss(gt_keypoints, pred_keypoints):
    loss = func.mse_loss(pred_keypoints, gt_keypoints)
    
    return loss

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot
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

# Define the model
model = KeypointPipeline(weights_path)
model = model.to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50 # Define your number of epochs
batch_size = 8

split_folder_path = train_test_split(root_dir)

KEYPOINTS_FOLDER_TRAIN = split_folder_path +"/train" #train_test_split(root_dir) +"/train"
KEYPOINTS_FOLDER_VAL = split_folder_path +"/val"
KEYPOINTS_FOLDER_TEST = split_folder_path +"/test"

dataset_train = KPDataset(KEYPOINTS_FOLDER_TRAIN, transform=None, demo=False)
dataset_val = KPDataset(KEYPOINTS_FOLDER_VAL, transform=None, demo=False)
dataset_test = KPDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)

data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

v = 0

model.train()
total_start_time = datetime.now()
loss_history = []
for epoch in range(num_epochs):
    epoch_start_time = datetime.now()  # Track start time for each epoch
    epoch_loss = 0
    for i, batch in enumerate(data_loader_train):
        img_tuple, target_dict_tuple, img_files = batch
#         print(f"Processing batch {i+1} with images:", img_files)
        print(f"\rProcessing epoch {epoch + 1}/{num_epochs}, batch {i + 1}/{len(data_loader_train)} with images: {img_files}", end="")
        
        imgs = [img.to(device) for img in img_tuple]  # Create list of images

        # Process each image individually
        losses = []
        for i in range(len(imgs)):
            img = imgs[i]  # Unsqueeze to add batch dimension

            # Prepare ground truth vertices for the image
            keypoints = target_dict_tuple[i]['keypoints'].to(device)
            # Process keypoints to get the format needed for loss computation
            keypoints_gt, valid_vis_all, valid_invis_all = process_keypoints([keypoints])             
            print("keypoints_gt", keypoints)
            print("valid_vis_all", valid_vis_all)
            print("valid_invis_all", valid_invis_all)

            # Forward pass
            output = model(img)
            edge_logits = output[0]
            keypoints_pred = output[1]
            init_keypoints = output[2]
            # Extract the first two coordinates (x, y) which are normalized
            normalized_keypoints = keypoints_pred[0, :, :2]
            

            # Denormalize the x coordinates
            denormalized_x = (normalized_keypoints[:, 0] * (640 / 2)) + (640 / 2)

            # Denormalize the y coordinates
            denormalized_y = (normalized_keypoints[:, 1] * (480 / 2)) + (480 / 2)

            # Stack the denormalized x and y coordinates together to form [n, 2] tensor
            denormalized_keypoints = torch.stack((denormalized_x, denormalized_y), dim=1)
            print("predicted keypoints", denormalized_keypoints)
            # Compute loss for the image
#             init_kp_loss = loss_kp(keypoints, init_keypoints)
            edge_loss = loss_edges(valid_vis_all, edge_logits)
            loss_kgnn2d = kgnn2d_loss(keypoints_gt, denormalized_keypoints)

            loss = edge_loss + loss_kgnn2d
            losses.append(loss)  # Store loss for the image
            
        # Average loss over all images in the batch
        total_loss = torch.mean(torch.stack(losses))
        epoch_loss += total_loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    epoch_avg_loss = epoch_loss / len(data_loader_train)
    loss_history.append(epoch_avg_loss)
    # Calculate time elapsed for the epoch
    epoch_end_time = datetime.now()
    epoch_time = (epoch_end_time - epoch_start_time).total_seconds()
    total_time_elapsed = (epoch_end_time - total_start_time).total_seconds()
    average_time_per_epoch = total_time_elapsed / (epoch + 1)
    eta = average_time_per_epoch * (num_epochs - epoch - 1)
    
    print(f'\nEpoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item()}, Epoch Time: {epoch_time}s, ETA: {eta:.2f}s')

    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss_history': loss_history
        }
        checkpoint_path = f"/home/schatterjee/lama/kprcnn_panda/trained_models/checkpoints/occ_net__epoch_{epoch + 1}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

model_save_path = f"/home/schatterjee/lama/kprcnn_panda/trained_models/krcnn_occ_b{batch_size}_e{num_epochs}_v{v}.pth"

torch.save(model, model_save_path)


