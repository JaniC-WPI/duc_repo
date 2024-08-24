#!/usr/bin/env python
# coding: utf-8

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
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


class KpVelDataset(Dataset):
    def __init__(self, json_folder):
        super(KpVelDataset, self).__init__()
        self.data = []
        for json_file in sorted(os.listdir(json_folder)):
            if json_file.endswith('_combined.json'):
                with open(os.path.join(json_folder, json_file), 'r') as file:
                    data = json.load(file)
                    start_kp = data['start_kp']
                    next_kp = data['next_kp']
                    position = data['position']
                    self.data.append((start_kp, next_kp, position))
    
    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     start_kp, next_kp, position = self.data[idx]
    #     # Ensure start_kp and next_kp have consistent dimensions
    #     # if not start_kp or not next_kp:
    #     #     raise ValueError(f"Empty keypoints found at index {idx}")
    #     start_kp_flat = torch.tensor([kp for sublist in start_kp for kp in sublist[0][:2]], dtype=torch.float)
    #     next_kp_flat = torch.tensor([kp for sublist in next_kp for kp in sublist[0][:2]], dtype=torch.float)
    #     position = torch.tensor(position, dtype=torch.float)

    #     return start_kp_flat, next_kp_flat, position

    def __getitem__(self, idx):
        start_kp, next_kp, position = self.data[idx]
        try:
            start_kp_flat = torch.tensor([kp for sublist in start_kp for kp in sublist[0][:2]], dtype=torch.float)
            next_kp_flat = torch.tensor([kp for sublist in next_kp for kp in sublist[0][:2]], dtype=torch.float)
            position = torch.tensor(position, dtype=torch.float)
            if start_kp_flat.nelement() != 18 or next_kp_flat.nelement() != 18:
                raise ValueError(f"Invalid number of elements: start_kp {start_kp_flat.nelement()}, next_kp {next_kp_flat.nelement()} at index {idx}")
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            print(f"Start KP: {start_kp}, Next KP: {next_kp}")
            raise
        return start_kp_flat, next_kp_flat, position

def train_test_split(src_dir):
    dst_dir_anno = src_dir + "annotations"
    
    if os.path.exists(dst_dir_anno):
        print("folders exist")
    else:
        os.mkdir(dst_dir_anno)

    for jsonfile in glob.iglob(os.path.join(src_dir, "*_combined.json")):
        shutil.copy(jsonfile, dst_dir_anno)
        
    output = root_dir + "split_folder_reg"
    
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

class PosRegModel(nn.Module):
    def __init__(self, input_size):
        super(PosRegModel, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 3)

    def forward(self, start_kp, next_kp):
        x = torch.cat((start_kp, next_kp), dim=1)
        x = func.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = func.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = func.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x
    
# def evaluate_model(model, data_loader, criterion):
#     model.to(device)
#     model.eval()
#     total_mse = 0
#     total_mae = 0
#     total_count = 0

#     with torch.no_grad():
#         for start_kp, next_kp, position in data_loader:
#             output = model(start_kp.to(device), next_kp.to(device))
#             mse_loss = criterion(output, position.to(device))
#             mae_loss = torch.nn.functional.l1_loss(output, position.to(device), reduction='sum')

#             total_mse += mse_loss.item() * position.size(0)
#             total_mae += mae_loss.item()
#             total_count += position.size(0)

#     avg_mse = total_mse / total_count
#     avg_rmse = np.sqrt(avg_mse)
#     avg_mae = total_mae / total_count

#     return avg_mse, avg_rmse, avg_mae

# def plot_and_save_train_loss_metrics(train_mse_loss_history, train_rmse_loss_history, train_mae_loss_history, b_size, num_epochs, v):
#     epochs = range(1, len(train_mse_loss_history) + 1)
#     plt.figure(figsize=(48, 8))
#     plt.subplot(3, 1, 1)
#     plt.plot(epochs, train_mse_loss_history, label='Training MSE')
#     plt.title(f'Training MSE\nBatch: {b_size}, Epochs: {num_epochs}')
#     plt.xlabel('Epochs')
#     plt.ylabel('MSE')
#     plt.legend()
#     plt.grid(True)
    
#     plt.subplot(3, 1, 2)
#     plt.plot(epochs, train_rmse_loss_history, label='Training RMSE', color='orange')
#     plt.title(f'Training RMSE\nBatch: {b_size}, Epochs: {num_epochs}')
#     plt.xlabel('Epochs')
#     plt.ylabel('RMSE')
#     plt.legend()
#     plt.grid(True)
    
#     plt.subplot(3, 1, 3)
#     plt.plot(epochs, train_mae_loss_history, label='Training MAE', color='green')
#     plt.title(f'Training MAE\nBatch: {b_size}, Epochs: {num_epochs}')
#     plt.xlabel('Epochs')
#     plt.ylabel('MAE')
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.savefig(f'/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/loss_plots/train_metrics_b{b_size}_e{num_epochs}_v{v}.png')
#     # plt.show()

# def plot_and_save_val_loss_metrics(val_mse_loss_history, val_rmse_loss_history, val_mae_loss_history, b_size, num_epochs, v):
#     epochs = range(1, len(val_mse_loss_history) + 1)
#     plt.figure(figsize=(48, 8))
#     plt.subplot(3, 1, 1)
#     plt.plot(epochs, val_mse_loss_history, label='Validation MSE')
#     plt.title(f'Validation MSE\nBatch: {b_size}, Epochs: {num_epochs}')
#     plt.xlabel('Epochs')
#     plt.ylabel('MSE')
#     plt.legend()
#     plt.grid(True)
    
#     plt.subplot(3, 1, 2)
#     plt.plot(epochs, val_rmse_loss_history, label='Validation RMSE', color='orange')
#     plt.title(f'Validation RMSE\nBatch: {b_size}, Epochs: {num_epochs}')
#     plt.xlabel('Epochs')
#     plt.ylabel('RMSE')
#     plt.legend()
#     plt.grid(True)
    
#     plt.subplot(3, 1, 3)
#     plt.plot(epochs, val_mae_loss_history, label='Validation MAE', color='green')
#     plt.title(f'Validation MAE\nBatch: {b_size}, Epochs: {num_epochs}')
#     plt.xlabel('Epochs')
#     plt.ylabel('MAE')
#     plt.legend()
#     plt.grid(True)
    
#     plt.tight_layout()
#     plt.savefig(f'/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/loss_plots/val_metrics_b{b_size}_e{num_epochs}_v{v}.png')
#     # plt.show()

# # # Initialize dataset and data loader
# # # to generalize home directory. User can change their parent path without entering their home directory
# epoch_list = [400,500,600]
# batch_sizes = [32,64,128]
# v = 11
# # parent_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/'
# root_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/regression_rearranged/'
# print(root_dir)
# split_folder_path = train_test_split(root_dir)

# print("where the split_folder is saved", split_folder_path)
# train_folder = split_folder_path + '/train/' + 'annotations/'
# val_folder = split_folder_path + '/val/' + 'annotations/'
# print("Train Folder:", train_folder)
# print("Validation Folder:", val_folder)
# train_dataset = KpVelDataset(train_folder)
# print(len(train_dataset))
# val_dataset = KpVelDataset(val_folder)

# summary_results = []
# for b_size in batch_sizes:
#     for num_epochs in epoch_list:
#         print(f'Epoch {num_epochs} and Batch {b_size}')
#         train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=False)
#         # Initialize model
#         model = PosRegModel(18)  # Adjust input_size as necessary
#         model.to(device)
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.005)
#         # Training loop
#         train_mse_loss_history = []
#         train_rmse_loss_history = []
#         train_mae_loss_history = []
#         val_mse_loss_history = []
#         val_rmse_loss_history = []
#         val_mae_loss_history = []

#         for epoch in range(num_epochs):
#             model.train()
#             for start_kp, next_kp, position in train_loader:
#                 optimizer.zero_grad()
#                 position = position.squeeze(1)
#                 # print(position)
#                 # print(start_kp.shape)
#                 # print(position.shape)
#                 output = model(start_kp.to(device), next_kp.to(device))
#                 loss = criterion(output, position.to(device))
#                 loss.backward()
#                 optimizer.step()
    
#             # Evaluate on the training set
#             train_mse, train_rmse, train_mae = evaluate_model(model, train_loader, criterion)
#             train_mse_loss_history.append(train_mse)
#             train_rmse_loss_history.append(train_rmse)
#             train_mae_loss_history.append(train_mae)
        
#             # Validation Loop
#             with torch.no_grad():
#                 val_loss = 0.0
#                 for start_kp, next_kp, position in val_loader: 
#                     output = model(start_kp.to(device), next_kp.to(device))
#                     loss = criterion(output, position.to(device))
#                     val_loss += loss.item()
                
#                 # Evaluate on the validation set
#                 val_mse, val_rmse, val_mae = evaluate_model(model, val_loader, criterion)
#                 val_mse_loss_history.append(val_mse)
#                 val_rmse_loss_history.append(val_rmse)
#                 val_mae_loss_history.append(val_mae)
        
#             # avg_val_loss = val_loss / len(val_loader)
#             # val_loss_history.append(avg_val_loss)
#                 # print("output", output)
#             # print(f'Epoch {epoch+1}, Loss: {loss.item()}')
#             print(f'Epoch {epoch+1}, Training MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}; Validation MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}')

#         summary = {
#                 'Batch Size': b_size,
#                 'Epochs': num_epochs,
#                 'Average Training MSE': np.mean(train_mse_loss_history),
#                 'Final Training MSE': train_mse_loss_history[-1],
#                 'Average Validation MSE': np.mean(val_mse_loss_history),
#                 'Final Validation MSE': val_mse_loss_history[-1],
#                 'Average Training RMSE': np.mean(train_rmse_loss_history),
#                 'Final Training RMSE': train_rmse_loss_history[-1],
#                 'Average Validation RMSE': np.mean(val_rmse_loss_history),
#                 'Final Validation RMSE': val_rmse_loss_history[-1],
#                 'Average Training MAE': np.mean(train_mae_loss_history),
#                 'Final Training MAE': train_mae_loss_history[-1],
#                 'Average Validation MAE': np.mean(val_mae_loss_history),
#                 'Final Validation MAE': val_mae_loss_history[-1]
#             }
        
#         summary_results.append(summary)
        
#         # # Save the trained model
#         model_save_path = f'/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b{b_size}_e{num_epochs}_v{v}.pth'
#         torch.save(model.state_dict(), model_save_path)

#         df_summary = pd.DataFrame(summary_results)

#         df_summary.to_csv('/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/loss_plots/summary_results.csv', index=False)


#         plot_and_save_train_loss_metrics(train_mse_loss_history, train_rmse_loss_history, train_mae_loss_history, b_size, num_epochs, v)
#         plot_and_save_val_loss_metrics(val_mse_loss_history, val_rmse_loss_history, val_mae_loss_history, b_size, num_epochs, v)

#         v += 1

# plt.plot(val_loss_history, label='Validation Loss') 
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Validation Loss Curve')
# plt.grid(True) 
# plt.legend() 
# plt.show()

# # model_save_path = f'/home/jc-merlab/Pictures/Data/trained_models/reg_nkp_b{batch_size}_e{num_epochs}_v{v}.pth'
# # torch.save({
# #     'model_state_dict': model.state_dict(),
# #     'model_structure': KeypointRegressionNet()
# # }, model_save_path)
# # print(f"Model saved to {model_save_path}")

# def test_model(model_path, test_data_dir):
#     # Load the test dataset
#     test_dataset = KpVelDataset(test_data_dir)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#     # Initialize the model and load the saved state
#     model = PosRegModel(12)    
#     model.load_state_dict(torch.load(model_path))
#     model.to(device)
#     model.eval()

#     # Criterion for evaluation
#     criterion = nn.MSELoss()
#     total_loss = 0

#     # No gradient needed for evaluation
#     with torch.no_grad():
#         for start_kp, next_kp, position in test_loader:
#             output = model(start_kp.to(device), next_kp.to(device))
#             for i in range(start_kp.size(0)):
#                 individual_start_kp = start_kp[i]
#                 individual_next_kp = next_kp[i]
#                 individual_position = position[i]
#                 predicted_position = output[i]

#                 print("Start KP:", individual_start_kp)
#                 print("Next KP:", individual_next_kp)
#                 print("Actual Position:", individual_position)
#                 print("Predicted Position:", predicted_position)
#                 print("-----------------------------------------")
#             loss = criterion(output, predicted_position)
#             total_loss += loss.item()
            
#     # Calculate the average loss
#     avg_loss = total_loss / len(test_loader)
#     print(f'Average Test Loss: {avg_loss}')

# # Usage
# model_path = '/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b64_e200_v1.pth'  # Update with your model path
# test_data_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/regression_combined_test_new/split_folder_reg/test/annotations/'  # Update with your test data path
# test_model(model_path, test_data_dir)
