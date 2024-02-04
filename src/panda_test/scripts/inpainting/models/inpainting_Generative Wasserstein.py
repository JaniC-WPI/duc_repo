#!/usr/bin/env python

from __future__ import print_function
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import os
import numpy as np
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
from models.dcgan import dcgan
import torch
import torch.optim
from sklearn.model_selection import train_test_split
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import TotalVariation

from utils_1.inpainting_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

PLOT = True
imsize = -1
dim_div_by = 64


train_path = '/home/jc-merlab/Venk/Augmented_Data/images'
label_path = '/home/jc-merlab/Venk/Augmented_Data/labels'

# img_list = []
# for images in os.listdir(train_path):
#     if (images.endswith(".jpg")):
#         img_list.append(images)
        

# img_list_sorted = sorted(img_list)


# lbl_list = []
# for images in os.listdir(label_path):
#     if (images.endswith(".jpg")):
#         lbl_list.append(images)
        

# lbl_list_sorted = sorted(lbl_list)



# # data_list = []
# # for file in os.listdir(train_path):

# #     #If file is a json, construct it's full path and open it, append all json data to list
# #     if file.endswith('json'):
# #         data_list.append(file)

# # data_list_sorted = sorted(data_list)
# # print(data_list_sorted)
# # print(len(data_list_sorted))

# print(img_list_sorted)
# print(len(img_list_sorted))
# print(lbl_list_sorted)
# print(len(lbl_list_sorted))

dtype = torch.float32
net = UNet()
net = net.type(dtype)



class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),  # input is 3 x 256 x 256
            *discriminator_block(64, 128),                                # output is 128 x 128 x 128
            *discriminator_block(128, 256),                               # output is 256 x 64 x 64
            *discriminator_block(256, 512),
            *discriminator_block(512, 1024),                                # output is 512 x 32 x 32
            nn.ZeroPad2d((1, 0, 1, 0)),                                   # Padding to match size
            nn.Conv2d(1024, 1, 4, padding=1)                               # Final output is 1 x 16 x 16
        )
        self.flatten = nn.Flatten()
        self.final = nn.Linear(15*20, 1)  # Adjust the input features to match the flattened size
        self.sigmoid = nn.Sigmoid()
    def forward(self, img):
        output = self.model(img)
        output = self.flatten(output)
        output = self.final(output)
        return output   #self.sigmoid(output)

discriminator = Discriminator()


input_tensor = torch.randn(8, 3, 480, 640)

# Getting the discriminator output for the input tensor
output = discriminator(input_tensor)
print(output.shape)


import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import random_split
import torch.optim as optim
import random



# class ImageLabelDataset(Dataset):
#     def __init__(self, image_dir, label_dir, transform=None):
#         """
#         Args:
#             image_dir (string): Directory with all the images.
#             label_dir (string): Directory with all the labels.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.image_dir = image_dir
#         self.label_dir = label_dir
#         self.image_files = os.listdir(image_dir)
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.image_dir, self.image_files[idx])
#         label_name = os.path.join(self.label_dir, self.image_files[idx])
        
#         image = Image.open(img_name)
#         label = Image.open(label_name)

#         if self.transform:
#             torch.manual_seed(0)
#             image = self.transform(image)
#             label = self.transform(label)

#         return image, label

# transform = transforms.Compose([
#     transforms.Resize((480, 640)),  
#     transforms.RandomHorizontalFlip(p=0.25),
#     transforms.RandomRotation(degrees=(0, 20)),
#     transforms.RandomVerticalFlip(p=0.25),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
#     transforms.ToTensor()

# ])

def denormalize(tensor, means, stds):
    means = torch.tensor(means).view(-1, 1, 1)
    stds = torch.tensor(stds).view(-1, 1, 1)

    if tensor.is_cuda:
        means = means.to(tensor.device)
        stds = stds.to(tensor.device)
    tensor = tensor * stds + means

    return tensor

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
        label = Image.open(label_name).convert('RGB')  # Assuming label is grayscale

        seed = np.random.randint(2147483647)  # make a seed with numpy generator 
        
        if self.image_transform or self.label_transform:
            random.seed(seed)  # apply this seed to img transforms
            torch.manual_seed(seed)
            image = self.image_transform(image)

            random.seed(seed)  # apply this seed to label transforms
            torch.manual_seed(seed)
            label = self.label_transform(label)

        return image, label

image_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.RandomHorizontalFlip(p=0.1),
    # transforms.RandomRotation(degrees=(-5, 5)),
    # # transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1),
    # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    transforms.RandomEqualize(p=0.01),
    transforms.RandomAutocontrast(p=0.01),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.209555113170385, 0.22507974363977162, 0.20982026500023962], std=[0.20639409678896012, 0.19208633033458372, 0.20659148273508857]),
])

label_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.RandomHorizontalFlip(p=0.1),
    # transforms.RandomRotation(degrees=(-5, 5)),
    # transforms.RandomVerticalFlip(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.209555113170385, 0.22507974363977162, 0.20982026500023962], std=[0.20639409678896012, 0.19208633033458372, 0.20659148273508857]),
])

image_dir = train_path
label_dir = label_path
dataset = ImageLabelDataset(image_dir=image_dir, label_dir=label_dir, 
                            image_transform=image_transform, 
                            label_transform=label_transform)

total_size = len(dataset)
train_size = int(0.9 * total_size)
val_size = (total_size - train_size) // 2
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

s  = sum(np.prod(list(p.size())) for p in net.parameters())
print ('Number of params: %d' % s)

mse = torch.nn.MSELoss().type(dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)


# for batch_idx, (data, target) in enumerate(train_loader):
#     if batch_idx < 2:    
#         # print(data.shape, target.shape)
#         data, target = data.to(device), target.to(device)
#         idx = np.random.randint(0, len(data))
#         print(data.shape, target.shape)
#         print(data[idx,:,:,:].permute(1,2,0).shape)
#         inp = data[idx,:,:,:].permute(1,2,0).detach().cpu().clone().numpy()
#         # out = output[idx,:,:,:].permute(1,2,0).detach().cpu().clone().numpy()
#         lab = target[idx,:,:,:].permute(1,2,0).detach().cpu().clone().numpy()
#         print(lab.shape, type(lab))
#         data = denormalize(data, [0.209555113170385, 0.22507974363977162, 0.20982026500023962], [0.20639409678896012, 0.19208633033458372, 0.20659148273508857] )
#         target = denormalize(target,[0.209555113170385, 0.22507974363977162, 0.20982026500023962], [0.20639409678896012, 0.19208633033458372, 0.20659148273508857])
#         for i, (j, k) in enumerate(zip(data, target)):
        
#             plt.figure(figsize=(10, 5))

#             ax1 = plt.subplot(1, 2, 1)
#             # print(i.shape, j.shape, k.shape)
#             plt.imshow(j.permute(1,2,0).detach().cpu().clone().numpy())
#             ax1.set_title('Input')
#             ax1.axis('off')


#             ax3 = plt.subplot(1, 2, 2)
#             plt.imshow(k.permute(1,2,0).detach().cpu().clone().numpy())
#             ax3.set_title('Target')
#             ax3.axis('off') 

#             plt.tight_layout()
#             plt.show()

#     else:
#         break
        



from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()



def wasserstein_loss(predictions, is_real):
    return predictions.mean() if is_real else -predictions.mean()

def discriminator_loss(real_predictions, fake_predictions):
    return fake_predictions.mean() - real_predictions.mean()

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))  #.requires_grad_(True)
    interpolates.requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# betas = (0.9, 0.95)
# batch_size = 16

criterion = nn.BCEWithLogitsLoss().type(dtype) 
reconstruction_criterion = nn.MSELoss().type(dtype) 

optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001) #  , weight_decay = 0.0001) #,betas= betas , lr = 0.0001
"""weight decay = 0.001 is tried."""
# optimizer_unet = torch.optim.Adam(unet.parameters(), lr=lr, betas=betas)
optimizer_g = optim.Adam(net.parameters(), lr=0.0001) #, weight_decay = 0.0001) #,betas= betas
real_label = 0.9
fake_label = 0.1
# lambda_adv = 0.00001
lambda_adv = 0.000001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
discriminator.to(device)


# from lpips import LPIPS
# from pytorch_fid import fid_score

# lpips_fn = LPIPS(net='alex').to(device)

# lpips_scores, fid_scores, tv_scores = [], [], []

# def calculate_tv(img):
#     """
#     Compute the Total Variation (TV) of an image.
#     Args:
#         img (Tensor): The image tensor. Shape (N, C, H, W).
#     Returns:
#         float: The total variation of the image.
#     """
#     batch_size, _, h, w = img.size()
#     tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
#     tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
#     return (tv_h + tv_w) / (batch_size * h * w)



# trained_model_final_unet_before_augmentation = '/home/jc-merlab/Venk/deep-image-prior-master/saved models/13Dec_att_generative_was_YCB.pth'
# net.load_state_dict(torch.load(trained_model_final_unet_before_augmentation))
# # trained_model_23_nov = '/home/jc-merlab/Venk/deep-image-prior-master/saved models/24nov_aug.pth'
# # net.load_state_dict(torch.load(trained_model_23_nov))


# # In[ ]:


# CHECKPOINT_PATH = '/home/jc-merlab/Venk/DIP_augmentation_2/Checkpoints'

# lambda_gp = 10
# # lambda_gp = 10
# # lambda_gp = 100 
# for epoch in range(10):
#     print("Epoch",epoch+1)
#     net.train()
#     discriminator.train()
#     d_loss = 0
#     g_loss = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         # print("Batch",batch_idx)
#         data, target = data.to(device), target.to(device)
#         fake_images = net(data)
#         real_images = target
        
#         optimizer_d.zero_grad()
#         real_predictions = discriminator(real_images)
#         fake_predictions = discriminator(fake_images.detach())
#         d_loss_real = wasserstein_loss(real_predictions, True)
#         d_loss_fake = wasserstein_loss(fake_predictions, False)
        
#         gradient_penalty = compute_gradient_penalty(discriminator, real_images.data, fake_images.data)
#         d_loss_total = d_loss_real + d_loss_fake + lambda_gp * gradient_penalty
        
#         d_loss_total.backward()
#         optimizer_d.step()
        
#         optimizer_g.zero_grad()
#         fake_images = net(data)
#         trick_predictions = discriminator(fake_images)
#         g_loss_recon = reconstruction_criterion(fake_images, real_images)
#         g_loss_adv = -trick_predictions.mean()  
#         g_loss_total = g_loss_recon + lambda_adv * g_loss_adv
#         g_loss_total.backward()
#         optimizer_g.step()

#         d_loss += d_loss_total.item()
#         g_loss += g_loss_total.item()
#     idx = np.random.randint(0, len(data))
#     data = denormalize(data, [0.209555113170385, 0.22507974363977162, 0.20982026500023962], [0.20639409678896012, 0.19208633033458372, 0.20659148273508857] )
#     fake_images = denormalize(fake_images,[0.209555113170385, 0.22507974363977162, 0.20982026500023962], [0.20639409678896012, 0.19208633033458372, 0.20659148273508857])
#     inp = data[idx,:,:,:].permute(1,2,0).detach().cpu().clone().numpy()
#     out = fake_images[idx,:,:,:].permute(1,2,0).detach().cpu().clone().numpy()
#     lab = target[idx,:,:,:].permute(1,2,0).detach().cpu().clone().numpy()
#     # print(out.shape, type(out), lab.shape, type(lab))

#     plt.figure(figsize=(15, 15))

#     ax1 = plt.subplot(1, 2, 1)
#     plt.imshow(inp)
#     ax1.set_title('Input')
#     ax1.axis('off')

#     ax2 = plt.subplot(1, 2, 2)
#     plt.imshow(out)
#     ax2.set_title('Output')
#     ax2.axis('off')  

#     # ax3 = plt.subplot(1, 3, 3)
#     # plt.imshow(lab)
#     # ax3.set_title('Target')
#     # ax3.axis('off') 

#     plt.tight_layout()
#     plt.show()
    
#     checkpoint = {
#         'epoch': epoch + 1,
#         'generator state_dict': net.state_dict(),
#         'discriminator state dict': discriminator.state_dict(),
#         'generator optimizer': optimizer_g.state_dict(),
#         'discriminator optimizer': optimizer_d.state_dict(),
#         'generator train_loss': g_loss,
#         'discriminator train loss': d_loss
#     }
#     checkpoint_filename = os.path.join(CHECKPOINT_PATH, f'checkpoint_epoch_{epoch+10}.pth')
#     torch.save(checkpoint, checkpoint_filename)
#     print(f"Checkpoint saved to {checkpoint_filename}")

#     net.eval()
#     discriminator.eval()
#     val_d_loss = 0
#     val_g_loss = 0
#     val_loss, lpips_val, tv_val, fid_val = 0, 0, 0, 0
#     with torch.no_grad(): 
         
#         for batch_idx, (data, target) in enumerate(val_loader): 
#             data, target = data.to(device), target.to(device)
#             fake_images = net(data)
#             real_images = target

#             real_labels =0.9*torch.ones((real_images.size(0), 1), device=device)
#             fake_labels = 0.1*torch.ones((fake_images.size(0), 1), device=device)

#             real_predictions = discriminator(real_images)
#             fake_predictions = discriminator(fake_images)
#             d_loss_real = criterion(real_predictions, real_labels)
#             d_loss_fake = criterion(fake_predictions, fake_labels)
#             fake_images = fake_images.to(device)
#             real_images = real_images.to(device)
#             lpips_value = lpips_fn(fake_images, real_images).mean()
#             lpips_val += lpips_value.item()
#             tv_value = calculate_tv(fake_images)
#             tv_val += tv_value.item()
            
#             d_loss_total = d_loss_real + d_loss_fake

#             trick_labels = 0.9*torch.ones(fake_images.size(0), 1, device=device)
#             trick_predictions = discriminator(fake_images)
#             g_loss_recon = reconstruction_criterion(fake_images, real_images)
#             g_loss_adv = criterion(trick_predictions, trick_labels)
#             g_loss_total = g_loss_recon + lambda_adv * g_loss_adv

#             val_d_loss += d_loss_total.item()
#             val_g_loss += g_loss_total.item()
#     # net.eval()
#     # discriminator.eval()

#     # with torch.no_grad():  # No gradients needed for evaluation, which saves memory and computations
#     #     val_d_loss = 0
#     #     val_g_loss = 0
#     #     val_g_recon_loss = 0
#     #     val_g_adv_loss = 0
        
#     #     for batch_idx, (data, target) in enumerate(val_loader):  # Replace 'val_loader' with your validation data loader
#     #         data, target = data.to(device), target.to(device)
#     #         fake_images = net(data)
#     #         real_images = target

#     #         # Evaluate discriminator's performance
#     #         real_predictions = discriminator(real_images)
#     #         fake_predictions = discriminator(fake_images)
#     #         d_loss_real_val = wasserstein_loss(real_predictions, True)
#     #         d_loss_fake_val = wasserstein_loss(fake_predictions, False)
#     #         torch.set_grad_enabled(True)
#     #         gradient_penalty_val = compute_gradient_penalty(discriminator, real_images.data, fake_images.data)
#     #         torch.set_grad_enabled(False)
#     #         d_loss_total_val = d_loss_real_val + d_loss_fake_val + lambda_gp * gradient_penalty_val

#     #         # Evaluate generator's performance
#     #         trick_predictions_val = discriminator(fake_images)
#     #         g_loss_recon_val = reconstruction_criterion(fake_images, real_images)
#     #         g_loss_adv_val = -trick_predictions_val.mean()  
#     #         g_loss_total_val = g_loss_recon_val + lambda_adv * g_loss_adv_val

#     #         # Accumulate the validation losses
#     #         val_d_loss += d_loss_total_val.item()
#     #         val_g_loss += g_loss_total_val.item()
#     #         val_g_recon_loss += g_loss_recon_val.item()
#     #         val_g_adv_loss += g_loss_adv_val.item()
#     print("Validation:")
#     idx = np.random.randint(0, len(data))
#     data = denormalize(data, [0.2015333830875326, 0.23070823518400257, 0.22909179679415806], [0.18555881044550504, 0.19490512330453, 0.2298110065725662] )
#     fake_images = denormalize(fake_images,[0.2015333830875326, 0.23070823518400257, 0.22909179679415806], [0.18555881044550504, 0.19490512330453, 0.2298110065725662])    
#     inp = data[idx,:,:,:].permute(1,2,0).detach().cpu().clone().numpy()
#     out = fake_images[idx,:,:,:].permute(1,2,0).detach().cpu().clone().numpy()
#     lab = target[idx,:,:,:].permute(1,2,0).detach().cpu().clone().numpy()
#     # print(out.shape, type(out), lab.shape, type(lab))

#     plt.figure(figsize=(15, 15))

#     ax1 = plt.subplot(1, 2, 1)
#     plt.imshow(inp)
#     ax1.set_title('Input')
#     ax1.axis('off')

#     ax2 = plt.subplot(1, 2, 2)
#     plt.imshow(out)
#     ax2.set_title('Output')
#     ax2.axis('off')  

#     # ax3 = plt.subplot(1, 3, 3)
#     # plt.imshow(lab)
#     # ax3.set_title('Target')
#     # ax3.axis('off') 

#     plt.tight_layout()
#     plt.show()

#     val_d_loss /= len(val_loader)
#     val_g_loss /= len(val_loader)
#     lpips_val /= len(val_loader)
#     tv_val /= len(val_loader)
#     # fid_val /= len(val_loader)
#     lpips_scores.append(lpips_val)
#     tv_scores.append(tv_val)
#     # val_g_recon_loss /= len(val_loader)
#     # val_g_adv_loss /= len(val_loader)
#     writer.add_scalar("Discriminator Loss/Validation", val_d_loss, epoch)
#     writer.add_scalar("Generator Loss/Validation", val_g_loss, epoch)
#     writer.add_scalar("Metrics/LPIPS", lpips_val, epoch)
#     writer.add_scalar("Metrics/Total Variation", tv_val, epoch)
#     # writer.add_scalar("Discriminator Loss/Validation", val_g_recon_loss, epoch)
#     # writer.add_scalar("Generator Loss/Validation", val_g_adv_loss, epoch)
#     print(f'Validation Discriminator Loss: {val_d_loss}')
#     print(f'Validation Generator Loss: {val_g_loss}')
#     # print(f'Validation Generator Reconstruction Loss: {val_g_recon_loss}')
#     # print(f'Validation Generator Adversarial Loss: {val_g_adv_loss}')
#     d_loss /= len(train_loader)
#     g_loss /= len(train_loader)
#     writer.add_scalar("Discriminator Loss/Epoch", d_loss, epoch)
#     writer.add_scalar("Generator Loss/ Epoch", g_loss, epoch)
#     print(f'Discriminator Loss Epoch {epoch}: {d_loss}')
#     print(f'Generator Loss Epoch {epoch}: {g_loss}')


# # In[ ]:


# torch.save(net.state_dict(), '/home/jc-merlab/Venk/deep-image-prior-master/saved models/14Dec_att_generative_was_YCB.pth')
# torch.save(discriminator.state_dict(), '/home/jc-merlab/Venk/deep-image-prior-master/saved models/14Dec_att_discrimiator_was_YCB.pth')
"""
"""
# torch.save(net.state_dict(), '/home/jc-merlab/Venk/deep-image-prior-master/saved models/20_nov_evening_att_generative_was.pth')
# torch.save(discriminator.state_dict(), '/home/jc-merlab/Venk/deep-image-prior-master/saved models/20_nov_evening_discriminator_was.pth')
"""
"""
# torch.save(net.state_dict(), '/home/jc-merlab/Venk/deep-image-prior-master/saved models/24_nov_evening_att_generative_was.pth')
# torch.save(discriminator.state_dict(), '/home/jc-merlab/Venk/deep-image-prior-master/saved models/24_nov_evening_discriminator_was.pth')


# In[ ]:


# net.load_state_dict(torch.load('/home/jc-merlab/Venk/deep-image-prior-master/saved models/16_nov_evening_att_generative_was.pth')) #5_att_generative_was
# discriminator.load_state_dict(torch.load('/home/jc-merlab/Venk/deep-image-prior-master/saved models/16_nov_evening_discriminator_was.pth'))
"""
"""
# net.load_state_dict(torch.load('/home/jc-merlab/Venk/deep-image-prior-master/saved models/20_nov_evening_att_generative_was.pth')) #5_att_generative_was
# discriminator.load_state_dict(torch.load('/home/jc-merlab/Venk/deep-image-prior-master/saved models/20_nov_evening_discriminator_was.pth'))
"""
"""
net.load_state_dict(torch.load('/home/jc-merlab/Venk/deep-image-prior-master/saved models/14Dec_att_generative_was_YCB.pth')) #5_att_generative_was
# discriminator.load_state_dict(torch.load('/home/jc-merlab/Venk/deep-image-prior-master/saved models/14Dec_att_discrimiator_was_YCB.pth'))


# In[ ]:


import cv2
# p1 = '/home/jc-merlab/Venk/16Nov_test/KP_test_wasserstein/input/'
# p2 = '/home/jc-merlab/Venk/16Nov_test/KP_test_wasserstein/labels/'
# p3 = '/home/jc-merlab/Venk/16Nov_test/KP_test_wasserstein/inpainted/'


# In[ ]:


# Make sure to call net.eval() and discriminator.eval() to set dropout and batch normalization layers 
# to evaluation mode before running inference
net.eval()
#discriminator.eval()
m = 0

with torch.no_grad():  # No gradients needed for testing, which saves memory and computations
    test_d_loss = 0
    test_g_loss = 0
    test_g_recon_loss = 0
    test_g_adv_loss = 0
    
    for batch_idx, (data, target) in enumerate(test_loader):  # Replace 'test_loader' with your test data loader
        if batch_idx < 5:
            data, target = data.to(device), target.to(device)
            fake_images = net(data)
            real_images = target

            # Evaluate discriminator's performance
            real_predictions = discriminator(real_images)
            fake_predictions = discriminator(fake_images)
            d_loss_real_test = wasserstein_loss(real_predictions, True)
            d_loss_fake_test = wasserstein_loss(fake_predictions, False)
            d_loss_total_test = d_loss_real_test + d_loss_fake_test

            # Evaluate generator's performance
            trick_predictions_test = discriminator(fake_images)
            g_loss_recon_test = reconstruction_criterion(fake_images, real_images)
            g_loss_adv_test = -trick_predictions_test.mean()  
            g_loss_total_test = g_loss_recon_test + lambda_adv * g_loss_adv_test

            # Accumulate the test losses
            test_d_loss += d_loss_total_test.item()
            test_g_loss += g_loss_total_test.item()
            test_g_recon_loss += g_loss_recon_test.item()
            test_g_adv_loss += g_loss_adv_test.item()

        
            data = denormalize(data, [0.209555113170385, 0.22507974363977162, 0.20982026500023962], [0.20639409678896012, 0.19208633033458372, 0.20659148273508857] )
            fake_images = denormalize(fake_images,[0.209555113170385, 0.22507974363977162, 0.20982026500023962], [0.20639409678896012, 0.19208633033458372, 0.20659148273508857])
            inp = data[idx,:,:,:].permute(1,2,0).detach().cpu().clone().numpy()
            for h,(i, j, k) in enumerate(zip(fake_images, data, target)):
                if batch_idx == 0 and h == 0:
                    iterator = 0
                
                inp = j.permute(1,2,0).detach().cpu().clone().numpy()
                out = i.permute(1,2,0).detach().cpu().clone().numpy()
                lab = k.permute(1,2,0).detach().cpu().clone().numpy()
                # print(out.shape, type(out), lab.shape, type(lab))
                
                m = iterator + m + 1
                name = p1 + f'{m}.jpg'
                name2 = p2 + f'{m}.jpg'
                name3 = p3 + f'{m}.jpg'

                # print(name)
                # print(name2)
                # print(name3)
                # cv2.imwrite(name, inp*255)
                # cv2.imwrite(name2, lab*255)
                # cv2.imwrite(name3, out*255)
                plt.figure(figsize=(20, 20))

                ax1 = plt.subplot(1, 2, 1)
                plt.imshow(inp)
                ax1.set_title('Input')
                ax1.axis('off')

                ax2 = plt.subplot(1, 2, 2)
                plt.imshow(out)
                ax2.set_title('Output')
                ax2.axis('off')  

                # # ax3 = plt.subplot(1, 3, 3)
                # # plt.imshow(lab)
                # # ax3.set_title('Target')
                # # ax3.axis('off') 

                plt.tight_layout()
                plt.show()

        else:
            break


    # Calculate the average losses for the entire test set
    test_d_loss /= len(test_loader)
    test_g_loss /= len(test_loader)
    test_g_recon_loss /= len(test_loader)
    test_g_adv_loss /= len(test_loader)

    # Print the average test losses
    print(f'Test Discriminator Loss: {test_d_loss}')
    print(f'Test Generator Loss: {test_g_loss}')
    print(f'Test Generator Reconstruction Loss: {test_g_recon_loss}')
    print(f'Test Generator Adversarial Loss: {test_g_adv_loss}')


# Testing on unknown dataset. (The style of making of the dataset is different)

# In[ ]:


p = '/home/jc-merlab/Venk/panda_raw_with_occ/test_27Oct'

#021277.rgb.jpg

imglst = []
for i, img in enumerate(os.listdir(p)):
    # if i < 16:
        if (images.endswith(".jpg")):
            imglst.append(img)


imglst_sorted = sorted(imglst)
print(imglst_sorted, "\n", len(imglst_sorted))

testset = ImageLabelDataset(image_dir=p, label_dir=p, image_transform=image_transform, label_transform=label_transform)
test = DataLoader(testset, batch_size=8, shuffle=False)


# In[ ]:


net.eval()
discriminator.eval()

test_d_loss = 0
test_g_loss = 0

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test):
        data, target = data.to(device), target.to(device)
        fake_images = net(data)
        real_images = target

        real_labels = 0.9 * torch.ones((real_images.size(0), 1), device=device)
        fake_labels = 0.1 * torch.ones((fake_images.size(0), 1), device=device)

        real_predictions = discriminator(real_images)
        fake_predictions = discriminator(fake_images)
        d_loss_real = criterion(real_predictions, real_labels)
        d_loss_fake = criterion(fake_predictions, fake_labels)
        d_loss_total = d_loss_real + d_loss_fake

        trick_labels = 0.9 * torch.ones(fake_images.size(0), 1, device=device)
        trick_predictions = discriminator(fake_images)
        g_loss_recon = reconstruction_criterion(fake_images, real_images)
        g_loss_adv = criterion(trick_predictions, trick_labels)
        g_loss_total = g_loss_recon + lambda_adv * g_loss_adv

        test_d_loss += d_loss_total.item()
        test_g_loss += g_loss_total.item()
        data = denormalize(data, [0.209555113170385, 0.22507974363977162, 0.20982026500023962], [0.20639409678896012, 0.19208633033458372, 0.20659148273508857] )
        fake_images = denormalize(fake_images,[0.209555113170385, 0.22507974363977162, 0.20982026500023962], [0.20639409678896012, 0.19208633033458372, 0.20659148273508857])
        inp = data[idx,:,:,:].permute(1,2,0).detach().cpu().clone().numpy()
        out = fake_images[idx,:,:,:].permute(1,2,0).detach().cpu().clone().numpy()
        lab = target[idx,:,:,:].permute(1,2,0).detach().cpu().clone().numpy()
        # print(out.shape, type(out), lab.shape, type(lab))

        plt.figure(figsize=(20, 20))

        ax1 = plt.subplot(1, 2, 1)
        plt.imshow(inp)
        ax1.set_title('Input')
        ax1.axis('off')

        ax2 = plt.subplot(1, 2, 2)
        plt.imshow(out)
        ax2.set_title('Output')
        ax2.axis('off')  

        # ax3 = plt.subplot(1, 3, 3)
        # plt.imshow(lab)
        # ax3.set_title('Target')
        # ax3.axis('off') 

        plt.tight_layout()
        plt.show()

        test_d_loss /= len(data)
        test_g_loss /= len(data)

        print(f'Test Discriminator Loss: {test_d_loss}')
        print(f'Test Generator Loss: {test_g_loss}')


# jnegierbgiebrig

# In[ ]:


import cv2
p1 = '/home/jc-merlab/Venk/Nov_11_2023_Testing/inpaint/images/'
p2 = '/home/jc-merlab/Venk/Nov_11_2023_Testing/inpaint/labels/'
dest = '/home/jc-merlab/Venk/Nov_11_2023_Testing/detect_kps/images/'


# In[ ]:


dataset = ImageLabelDataset(image_dir=p1, label_dir=p2, image_transform=image_transform, label_transform=label_transform)
test_loader = DataLoader(dataset, batch_size=12, shuffle=False)


# In[ ]:


net.eval()
discriminator.eval()
m = 0
test_d_loss = 0
test_g_loss = 0

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx <5:
            data, target = data.to(device), target.to(device)
            fake_images = net(data)
            real_images = target

            real_labels = 0.9 * torch.ones((real_images.size(0), 1), device=device)
            fake_labels = 0.1 * torch.ones((fake_images.size(0), 1), device=device)

            real_predictions = discriminator(real_images)
            fake_predictions = discriminator(fake_images)
            d_loss_real = criterion(real_predictions, real_labels)
            d_loss_fake = criterion(fake_predictions, fake_labels)
            d_loss_total = d_loss_real + d_loss_fake

            trick_labels = 0.9 * torch.ones(fake_images.size(0), 1, device=device)
            trick_predictions = discriminator(fake_images)
            g_loss_recon = reconstruction_criterion(fake_images, real_images)
            g_loss_adv = criterion(trick_predictions, trick_labels)
            g_loss_total = g_loss_recon + lambda_adv * g_loss_adv

            test_d_loss += d_loss_total.item()
            test_g_loss += g_loss_total.item()
            data = denormalize(data, [0.209555113170385, 0.22507974363977162, 0.20982026500023962], [0.20639409678896012, 0.19208633033458372, 0.20659148273508857] )
            fake_images = denormalize(fake_images,[0.209555113170385, 0.22507974363977162, 0.20982026500023962], [0.20639409678896012, 0.19208633033458372, 0.20659148273508857])

            for h,(i, j, k) in enumerate(zip(fake_images, data, target)):
                if batch_idx == 0 and h == 0:
                    iterator = 0

                inp = j.permute(1,2,0).detach().cpu().clone().numpy()
                out = i.permute(1,2,0).detach().cpu().clone().numpy()
                lab = k.permute(1,2,0).detach().cpu().clone().numpy()
                # print(out.shape, type(out), lab.shape, type(lab))
                
                m = iterator + m + 1
                name = p1 + f'{m}.jpg'

                print(name)
                # cv2.imwrite(name, out*255)

                plt.figure(figsize=(20, 20))

                ax1 = plt.subplot(1, 2, 1)
                plt.imshow(inp)
                ax1.set_title('Input')
                ax1.axis('off')

                ax2 = plt.subplot(1, 2, 2)
                plt.imshow(out)
                ax2.set_title('Output')
                ax2.axis('off')  

                # ax3 = plt.subplot(1, 3, 3)
                # plt.imshow(lab)
                # ax3.set_title('Target')
                # ax3.axis('off') 

                plt.tight_layout()
                plt.show()


            test_d_loss /= len(data)
            test_g_loss /= len(data)

            print(f'Test Discriminator Loss: {test_d_loss}')
            print(f'Test Generator Loss: {test_g_loss}')
        else:
            break


# testing random sofa objects

# In[ ]:


import cv2
p1 = '/home/jc-merlab/Venk/New_Objects_Test_28_Nov/images'
p2 = '/home/jc-merlab/Venk/New_Objects_Test_28_Nov/labels'
# dest = '/home/jc-merlab/Venk/Nov_11_2023_Testing/detect_kps/images/'


# In[ ]:


d= ImageLabelDataset(image_dir=p1, label_dir=p2, image_transform=image_transform, label_transform=label_transform)
te = DataLoader(d, batch_size=4, shuffle=False)


# In[ ]:


for batch_idx, (data, target) in enumerate(te):
    if batch_idx < 2:    
        # print(data.shape, target.shape)
        data, target = data.to(device), target.to(device)
        idx = np.random.randint(0, len(data))
        print(data.shape, target.shape)
        print(data[idx,:,:,:].permute(1,2,0).shape)
        inp = data[idx,:,:,:].permute(1,2,0).detach().cpu().clone().numpy()
        # out = output[idx,:,:,:].permute(1,2,0).detach().cpu().clone().numpy()
        lab = target[idx,:,:,:].permute(1,2,0).detach().cpu().clone().numpy()
        print(lab.shape, type(lab))
        data = denormalize(data, [0.209555113170385, 0.22507974363977162, 0.20982026500023962], [0.20639409678896012, 0.19208633033458372, 0.20659148273508857] )
        target = denormalize(target,[0.209555113170385, 0.22507974363977162, 0.20982026500023962], [0.20639409678896012, 0.19208633033458372, 0.20659148273508857])
        for i, (j, k) in enumerate(zip(data, target)):
        
            plt.figure(figsize=(10, 5))

            ax1 = plt.subplot(1, 2, 1)
            # print(i.shape, j.shape, k.shape)
            plt.imshow(j.permute(1,2,0).detach().cpu().clone().numpy())
            ax1.set_title('Input')
            ax1.axis('off')


            ax3 = plt.subplot(1, 2, 2)
            plt.imshow(k.permute(1,2,0).detach().cpu().clone().numpy())
            ax3.set_title('Target')
            ax3.axis('off') 

            plt.tight_layout()
            plt.show()

    else:
        break
        


# In[ ]:


net.eval()
discriminator.eval()
m = 0
test_d_loss = 0
test_g_loss = 0

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(te):
        if batch_idx <5:
            data, target = data.to(device), target.to(device)
            fake_images = net(data)
            real_images = target

            real_labels = 0.9 * torch.ones((real_images.size(0), 1), device=device)
            fake_labels = 0.1 * torch.ones((fake_images.size(0), 1), device=device)

            real_predictions = discriminator(real_images)
            fake_predictions = discriminator(fake_images)
            d_loss_real = criterion(real_predictions, real_labels)
            d_loss_fake = criterion(fake_predictions, fake_labels)
            d_loss_total = d_loss_real + d_loss_fake

            trick_labels = 0.9 * torch.ones(fake_images.size(0), 1, device=device)
            trick_predictions = discriminator(fake_images)
            g_loss_recon = reconstruction_criterion(fake_images, real_images)
            g_loss_adv = criterion(trick_predictions, trick_labels)
            g_loss_total = g_loss_recon + lambda_adv * g_loss_adv

            test_d_loss += d_loss_total.item()
            test_g_loss += g_loss_total.item()
            data = denormalize(data, [0.209555113170385, 0.22507974363977162, 0.20982026500023962], [0.20639409678896012, 0.19208633033458372, 0.20659148273508857] )
            fake_images = denormalize(fake_images,[0.209555113170385, 0.22507974363977162, 0.20982026500023962], [0.20639409678896012, 0.19208633033458372, 0.20659148273508857])

            for h,(i, j, k) in enumerate(zip(fake_images, data, target)):
                if batch_idx == 0 and h == 0:
                    iterator = 0

                inp = j.permute(1,2,0).detach().cpu().clone().numpy()
                out = i.permute(1,2,0).detach().cpu().clone().numpy()
                lab = k.permute(1,2,0).detach().cpu().clone().numpy()
                # print(out.shape, type(out), lab.shape, type(lab))
                
                m = iterator + m + 1
                name = p1 + f'{m}.jpg'

                print(name)
                # cv2.imwrite(name, out*255)

                plt.figure(figsize=(20, 20))

                ax1 = plt.subplot(1, 2, 1)
                plt.imshow(inp)
                ax1.set_title('Input')
                ax1.axis('off')

                ax2 = plt.subplot(1, 2, 2)
                plt.imshow(out)
                ax2.set_title('Output')
                ax2.axis('off')  

                # ax3 = plt.subplot(1, 3, 3)
                # plt.imshow(lab)
                # ax3.set_title('Target')
                # ax3.axis('off') 

                plt.tight_layout()
                plt.show()


            test_d_loss /= len(data)
            test_g_loss /= len(data)

            print(f'Test Discriminator Loss: {test_d_loss}')
            print(f'Test Generator Loss: {test_g_loss}')
        else:
            break

