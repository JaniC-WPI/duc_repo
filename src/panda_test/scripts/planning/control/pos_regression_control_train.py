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
from sklearn.metrics import r2_score


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
    
def evaluate_model(model, data_loader, criterion):
    model.to(device)
    model.eval()
    total_mse = 0
    total_mae = 0
    total_count = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for start_kp, next_kp, position in data_loader:
            output = model(start_kp.to(device), next_kp.to(device))
            mse_loss = criterion(output, position.to(device))
            mae_loss = torch.nn.functional.l1_loss(output, position.to(device), reduction='sum')

            total_mse += mse_loss.item() * position.size(0)
            total_mae += mae_loss.item()
            total_count += position.size(0)

            all_preds.append(output.cpu().numpy())
            all_targets.append(position.cpu().numpy())

    avg_mse = total_mse / total_count
    avg_rmse = np.sqrt(avg_mse)
    avg_mae = total_mae / total_count

    # Flatten predictions and targets to calculate R²
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    r2 = r2_score(all_targets, all_preds)

    return avg_mse, avg_rmse, avg_mae, r2


def save_loss_data_to_csv(train_mse, train_rmse, train_mae, train_r2, val_mse, val_rmse, val_mae, val_r2, b_size, num_epochs, v):
    df = pd.DataFrame({
        'Epoch': range(1, len(train_mse) + 1),
        'Training MSE': train_mse,
        'Training RMSE': train_rmse,
        'Training MAE': train_mae,
        'Training R2': train_r2,
        'Validation MSE': val_mse,
        'Validation RMSE': val_rmse,
        'Validation MAE': val_mae,
        'Validation R2': val_r2
    })
    file_path = f'/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/loss_plots_recorrected/metrics_b{b_size}_e{num_epochs}_v{v}.csv'
    df.to_csv(file_path, index=False)
    print(f"Loss data saved to {file_path}")

def create_summary(train_mse, train_rmse, train_mae, train_r2, val_mse, val_rmse, val_mae, val_r2, b_size, num_epochs):
    summary = {
        'Batch Size': b_size,
        'Epochs': num_epochs,
        'Average Training MSE': np.mean(train_mse),
        'Final Training MSE': train_mse[-1],
        'Average Validation MSE': np.mean(val_mse),
        'Final Validation MSE': val_mse[-1],
        'Average Training RMSE': np.mean(train_rmse),
        'Final Training RMSE': train_rmse[-1],
        'Average Validation RMSE': np.mean(val_rmse),
        'Final Validation RMSE': val_rmse[-1],
        'Average Training MAE': np.mean(train_mae),
        'Final Training MAE': train_mae[-1],
        'Average Validation MAE': np.mean(val_mae),
        'Final Validation MAE': val_mae[-1],
        'Average Training R2': np.mean(train_r2),
        'Final Training R2': train_r2[-1],
        'Average Validation R2': np.mean(val_r2),
        'Final Validation R2': val_r2[-1]
    }
    return summary

def plot_loss_metrics(train_mse, train_rmse, train_mae, train_r2, val_mse, val_rmse, val_mae, val_r2, b_size, num_epochs, v):
    epochs = range(1, len(train_mse) + 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.plot(epochs, train_mse, label='Training MSE')
    plt.plot(epochs, val_mse, label='Validation MSE', linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 2)
    plt.plot(epochs, train_rmse, label='Training RMSE', color='orange')
    plt.plot(epochs, val_rmse, label='Validation RMSE', linestyle='dashed', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 3)
    plt.plot(epochs, train_mae, label='Training MAE', color='green')
    plt.plot(epochs, val_mae, label='Validation MAE', linestyle='dashed', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 4)
    plt.plot(epochs, train_r2, label='Training R2', color='blue')
    plt.plot(epochs, val_r2, label='Validation R2', linestyle='dashed', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('R2_Score')
    plt.legend()
    plt.grid(True)


    
    plt.tight_layout()
    plt.savefig(f'/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/loss_plots_recorrected/metrics_b{b_size}_e{num_epochs}_v{v}.png')
    print(f"Plot saved for batch size {b_size} and {num_epochs} epochs.")

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
#     plt.savefig(f'/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/loss_plots_half/train_metrics_b{b_size}_e{num_epochs}_v{v}.png')
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
#     plt.savefig(f'/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/loss_plots_half/val_metrics_b{b_size}_e{num_epochs}_v{v}.png')
#     # plt.show()

# # Initialize dataset and data loader
# # to generalize home directory. User can change their parent path without entering their home directory
epoch_list = [400,500,600]
batch_sizes = [128]
v = 32
# parent_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/'
root_dir = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/regression_rearranged_all_double_corrected/'
print(root_dir)
split_folder_path = train_test_split(root_dir)

print("where the split_folder is saved", split_folder_path)
train_folder = split_folder_path + '/train/' + 'annotations/'
val_folder = split_folder_path + '/val/' + 'annotations/'
print("Train Folder:", train_folder)
print("Validation Folder:", val_folder)
train_dataset = KpVelDataset(train_folder)
print(len(train_dataset))
val_dataset = KpVelDataset(val_folder)

summary_results = []
for b_size in batch_sizes:
    for num_epochs in epoch_list:
        print(f'Epoch {num_epochs} and Batch {b_size}')
        train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=False)
        # Initialize model
        model = PosRegModel(18)  # Adjust input_size as necessary
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        # Training loop
        train_mse_loss_history = []
        train_rmse_loss_history = []
        train_mae_loss_history = []
        train_r2_history = []
        val_mse_loss_history = []
        val_rmse_loss_history = []
        val_mae_loss_history = []
        val_r2_history = []

        for epoch in range(num_epochs):
            model.train()
            for start_kp, next_kp, position in train_loader:
                optimizer.zero_grad()
                position = position.squeeze(1)
                # print(position)
                # print(start_kp.shape)
                # print(position.shape)
                output = model(start_kp.to(device), next_kp.to(device))
                loss = criterion(output, position.to(device))
                loss.backward()
                optimizer.step()
    
            # Evaluate on the training set
            train_mse, train_rmse, train_mae, train_r2 = evaluate_model(model, train_loader, criterion)
            train_mse_loss_history.append(train_mse)
            train_rmse_loss_history.append(train_rmse)
            train_mae_loss_history.append(train_mae)
            train_r2_history.append(train_r2)
        
            # Validation Loop
            with torch.no_grad():
                val_loss = 0.0
                for start_kp, next_kp, position in val_loader: 
                    output = model(start_kp.to(device), next_kp.to(device))
                    loss = criterion(output, position.to(device))
                    val_loss += loss.item()
                
                # Evaluate on the validation set
                val_mse, val_rmse, val_mae, val_r2 = evaluate_model(model, val_loader, criterion)
                val_mse_loss_history.append(val_mse)
                val_rmse_loss_history.append(val_rmse)
                val_mae_loss_history.append(val_mae)
                val_r2_history.append(val_r2)
        
            # avg_val_loss = val_loss / len(val_loader)
            # val_loss_history.append(avg_val_loss)
                # print("output", output)
            # print(f'Epoch {epoch+1}, Loss: {loss.item()}')
            print(f'Epoch {epoch+1}, Training MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R2: {train_r2}; Validation MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2}')

        # Save detailed loss data and plot
        save_loss_data_to_csv(train_mse_loss_history, train_rmse_loss_history, train_mae_loss_history, train_r2_history,
                              val_mse_loss_history, val_rmse_loss_history, val_mae_loss_history, val_r2_history, b_size, num_epochs, v)
        plot_loss_metrics(train_mse_loss_history, train_rmse_loss_history, train_mae_loss_history, train_r2_history,
                          val_mse_loss_history, val_rmse_loss_history, val_mae_loss_history, val_r2_history, b_size, num_epochs, v)

        # Create and save summary
        summary = create_summary(train_mse_loss_history, train_rmse_loss_history, train_mae_loss_history, train_r2_history,
                                 val_mse_loss_history, val_rmse_loss_history, val_mae_loss_history, val_r2_history, b_size, num_epochs)
        summary_results.append(summary)
        # Save the trained model
        model_save_path = f'/home/jc-merlab/Pictures/Data/trained_models/reg_pos_b{b_size}_e{num_epochs}_v{v}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        v += 1  # Increment version for saving files

# Save summary to CSV
df_summary = pd.DataFrame(summary_results)
summary_path = '/home/jc-merlab/Pictures/panda_data/panda_sim_vel/panda_rearranged_data/loss_plots_recorrected/summary_results.csv'
df_summary.to_csv(summary_path, index=False)
print(f"Summary saved to {summary_path}")

