#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import Dataset
import json
import os
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader
import torch.optim as optim
from os.path import expanduser
import splitfolders
import shutil
import glob
import numpy as np
from sklearn.model_selection import train_test_split


# In[ ]:


class KpVelDataset(Dataset):
#     def __init__(self, json_folder):
#         super(KpVelDataset, self).__init__()
#         self.data = []
#         for json_file in sorted(os.listdir(json_folder)):
#             if json_file.endswith('_combined.json'):
#                 with open(os.path.join(json_folder, json_file), 'r') as file:
#                     data = json.load(file)
#                     start_kp = data['start_kp']
#                     next_kp = data['next_kp']
#                     position = data['position']
#                     self.data.append((start_kp, next_kp, position))

    def __init__(self, json_folder):
        super(KpVelDataset, self).__init__()
        self.data = []
        for json_file in sorted(os.listdir(json_folder)):
            if json_file.endswith('_combined.json'):
                with open(os.path.join(json_folder, json_file), 'r') as file:
                    data = json.load(file)
                    # Ensure data contains 'start_kp', 'next_kp', and 'position'
                    if 'start_kp' in data and 'next_kp' in data and 'position' in data:
                        start_kp = data['start_kp']
                        next_kp = data['next_kp']
                        position = data['position']
                        # Only append if start_kp and next_kp are not empty
                        if start_kp and next_kp:
                            self.data.append((start_kp, next_kp, position))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        start_kp, next_kp, position = self.data[idx]
        # Ensure start_kp and next_kp have consistent dimensions
        if not start_kp or not next_kp:
            raise ValueError(f"Empty keypoints found at index {idx}")
        start_kp_flat = torch.tensor([kp for sublist in start_kp for kp in sublist[0][:2]], dtype=torch.float)
        next_kp_flat = torch.tensor([kp for sublist in next_kp for kp in sublist[0][:2]], dtype=torch.float)
        position = torch.tensor(position, dtype=torch.float)
        return start_kp_flat, next_kp_flat, position

#     def __len__(self):
#         return len(self.data)

# #     def __getitem__(self, idx):
# #         start_kp, next_kp, velocity = self.data[idx]
# #         start_kp_flat = torch.tensor([kp for sublist in start_kp for kp in sublist[0]])
# #         next_kp_flat = torch.tensor([kp for sublist in next_kp for kp in sublist[0]])
# #         velocity = torch.tensor(velocity)
# #         return start_kp_flat, next_kp_flat, velocity
  
#     def __getitem__(self, idx):
#         start_kp, next_kp, position = self.data[idx]
#         # Extract and flatten the first two elements of each keypoint in start_kp
#         start_kp_flat = torch.tensor([kp for sublist in start_kp for kp in sublist[0][:2]], dtype=torch.float)
#         # Extract and flatten the first two elements of each keypoint in next_kp
#         next_kp_flat = torch.tensor([kp for sublist in next_kp for kp in sublist[0][:2]], dtype=torch.float)
#         position = torch.tensor(position)
#         return start_kp_flat, next_kp_flat, position
    


# In[ ]:


def train_test_split(src_dir):
#     dst_dir_img = src_dir + "images"
    dst_dir_anno = src_dir + "annotations"
    
    if os.path.exists(dst_dir_anno):
        print("folders exist")
    else:
        os.mkdir(dst_dir_anno)
        
#     for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
#         shutil.copy(jpgfile, dst_dir_img)

    for jsonfile in glob.iglob(os.path.join(src_dir, "*_combined.json")):
        shutil.copy(jsonfile, dst_dir_anno)
        
    output = src_dir + "split_folder_reg"
    
    splitfolders.ratio(src_dir, # The location of dataset
                   output=output, # The output location
                   seed=42, # The number of seed
                   ratio=(0.8, 0.1, 0.1), # The ratio of split dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
                   )
    
#     shutil.rmtree(dst_dir_img)
    shutil.rmtree(dst_dir_anno)
    
    return output  


# In[ ]:


class PosRegModel(nn.Module):
    def __init__(self, input_size):
        super(PosRegModel, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, 1024)  # Assuming start_kp and next_kp are concatenated
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,128)
        self.fc5 = nn.Linear(128,64)
        self.fc6 = nn.Linear(64,64)
        self.fc7 = nn.Linear(64,3)  # Output size is 3 for velocity

    def forward(self, start_kp, next_kp):
        x = torch.cat((start_kp, next_kp), dim=1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = func.relu(self.fc4(x))
        x = func.relu(self.fc5(x))
        x = func.relu(self.fc6(x))
        x = self.fc7(x)
        return x


# In[ ]:


# Initialize dataset and data loader
# to generalize home directory. User can change their parent path without entering their home directory
num_epochs = 300
batch_size = 64
v = 1
root_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/regression_combined_test_new/'
print(root_dir)
split_folder_path = train_test_split(root_dir)
dataset = KpVelDataset(root_dir)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = PosRegModel(12)  # Adjust input_size as necessary
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training loop
for epoch in range(num_epochs):
    for start_kp, next_kp, position in data_loader:
        optimizer.zero_grad()
        position = position.squeeze(1)
        print(position)
        print(start_kp.shape)
        print(position.shape)
        output = model(start_kp, next_kp)
        loss = criterion(output, position)
        loss.backward()
        optimizer.step()
        print("output", output)
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
# # Save the trained model
model_save_path = f'/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b{batch_size}_e{num_epochs}_v{v}.pth'
torch.save(model.state_dict(), model_save_path)

# model_save_path = f'/home/jc-merlab/Pictures/Data/trained_models/reg_nkp_b{batch_size}_e{num_epochs}_v{v}.pth'
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'model_structure': KeypointRegressionNet()
# }, model_save_path)
# print(f"Model saved to {model_save_path}")

