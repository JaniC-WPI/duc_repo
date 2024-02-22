#!/usr/bin/env python
from __future__ import print_function
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import numpy as np
import torch
import torch.optim
from sklearn.model_selection import train_test_split
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import TotalVariation
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
# import torchvision.transforms.functional as F
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import random
from torchvision.models import resnet50
from torch.nn import MSELoss
from torch.cuda.amp import GradScaler, autocast
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

PLOT = True
imsize = -1
dim_div_by = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

t = torch.cuda.get_device_properties(0).total_memory
print(t)
torch.cuda.empty_cache()

r = torch.cuda.memory_reserved(0)
print(r)
a = torch.cuda.memory_allocated(0)
print(a)
f = r-a  # free inside reserved

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

class ImageLabelDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_transform=None, label_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = os.listdir(image_dir)
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        label_name = os.path.join(self.label_dir, self.image_files[idx])

        image = Image.open(img_name).convert('RGB')
        label = Image.open(label_name).convert('RGB')

        seed = np.random.randint(2147483647)  # make a seed with numpy generator 
        
        if self.image_transform is not None:
            random.seed(seed)  # Apply this seed to image transforms
            torch.manual_seed(seed)
            image = self.image_transform(image)

        if self.label_transform is not None:
            random.seed(seed)  # Apply this seed to label transforms
            torch.manual_seed(seed)
            label = self.label_transform(label)

        return image, label



image_dir = '/home/jc-merlab/Pictures/panda_data/panda_vit_data/images/'
label_dir = '/home/jc-merlab/Pictures/panda_data/panda_vit_data/labels'


mean = torch.tensor([0.2367, 0.2567, 0.2429])
std = torch.tensor([0.2261, 0.2213, 0.2405])

image_transform = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1),
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    transforms.RandomEqualize(p=0.01),
    transforms.RandomAutocontrast(p=0.01),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

label_transform = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = ImageLabelDataset(image_dir=image_dir, label_dir=label_dir, 
                            image_transform=image_transform, 
                            label_transform=label_transform)

total_size = len(dataset)
train_size = int(0.9 * total_size)
val_size = (total_size - train_size) // 2
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor * std + mean
    return tensor

def show_image(img_tensor, mean, std, title="", normalize=True, original_size=(480, 640)):
    """
    Display a tensor as an image.
    """
    if normalize:
        denorm_img = denormalize(img_tensor, mean, std)
    
    img_tensor = F.interpolate(denorm_img.unsqueeze(0), size=original_size, mode='bilinear', align_corners=False).squeeze(0)
    # Convert tensor to numpy for visualization
    img = img_tensor.detach().cpu().numpy().transpose((1, 2, 0))  # Adjust for channel ordering
    img = np.clip(img, 0, 1)  # Ensure the image's pixel values are valid after denormalization
    
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

def visualize_dataset(dataset_loader):
    """
    Visualizes a batch of images and labels from the dataset loader.
    """
    # Fetch a batch of images and labels
    images, labels = next(iter(dataset_loader))
    
    batch_size = len(images)
    
    plt.figure(figsize=(10, 4 * batch_size))
    
    for idx in range(batch_size):
        plt.subplot(batch_size, 2, 2*idx + 1)
        show_image(images[idx], mean, std, title=f"Image {idx}")
        
        plt.subplot(batch_size, 2, 2*idx + 2)
        show_image(labels[idx], mean, std, title=f"Label {idx}")  # Assuming labels are already normalized
        
    plt.tight_layout()
    plt.show()

# Visualize a batch from the train_loader
# visualize_dataset(val_loader)

import torch
import torch.nn as nn
from torchvision.models import vit_h_14, ViT_H_14_Weights

class ViTInpainting(nn.Module):
    def __init__(self, image_size=224, channels=3):
        super(ViTInpainting, self).__init__()

        # Load the specific pre-trained ViT model
        weights = ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.vit = vit_h_14(weights=weights)

        # Decoder to reconstruct the image
        self.decoder = nn.Sequential(
            nn.Linear(1000, image_size * image_size * channels),
            nn.Unflatten(1, (channels, image_size, image_size)),
            nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Ensure output pixel values are between 0 and 1
        )
        
        # Replace the classification head with a dummy, if needed
        # Check the specific model structure for how to best do this

    def forward(self, x):
        # Pass input through ViT
        features = self.vit(x)
        
        # Decode features to reconstruct the image
        reconstructed_img = self.decoder(features)
        
        return reconstructed_img

reconstruction_loss = nn.L1Loss()
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.resnet50 = resnet50(pretrained=True).eval()
        # Remove the fully connected layer to get feature representations
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])
        for param in self.resnet50.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        # Ensure input tensors are detached and do not require gradients
        output_features = self.resnet50(output.detach())
        target_features = self.resnet50(target.detach())
        loss = F.l1_loss(output_features, target_features)
        return loss

model = ViTInpainting().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
reconstruction_criterion = nn.L1Loss()
perceptual_criterion = PerceptualLoss().to(device)

num_epochs = 2
scaler = GradScaler()
accumulation_steps = 4

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):  # Assuming 'dataloader' is defined
        if torch.cuda.is_available():
            inputs, targets = data  # 'inputs' are occluded images, 'targets' are full images
            inputs, targets = inputs.to(device), targets.to(device)        
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            recon_loss = reconstruction_criterion(outputs, targets)
            perceptual_loss = perceptual_criterion(outputs, targets)
            loss = (recon_loss + perceptual_loss) / accumulation_steps
        # Backward pass with scaled loss
        scaler.scale(loss).backward()  
        
        if (i + 1) % accumulation_steps == 0:  # Perform the optimization step every `accumulation_steps`
            scaler.step(optimizer)  # Update weights
            scaler.update()
            optimizer.zero_grad()  # Clear gradients
       
    # Checkpoint saving
    if (epoch + 1) % 10 == 0:
        checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')
        
# Save the final model
final_model_path = 'final_model.pth'
torch.save(model.state_dict(), final_model_path)
print(f'Saved final model to {final_model_path}')

