#!/usr/bin/env python3.8
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

class VelRegModel(nn.Module):
    def __init__(self, input_size):
        super(VelRegModel, self).__init__()
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