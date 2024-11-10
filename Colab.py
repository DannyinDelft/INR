# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:04:57 2024

@author: Gebruiker
"""

# -*- coding: utf-8 -*-
"""
Script with controlled dimensions, normalization, and fixed 30x30 resolution.
"""
!pip install rasterio
!pip install plyfile
!pip install configargparse

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rasterio
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Set device configuration
device = torch.device('cpu')
print(f"Running on device: {device}")

# Define a memory-efficient SIREN-based model using convolutions with controlled output
class CustomSirenModel(nn.Module):
    def __init__(self, in_channels=1, interm_channels=32, out_channels=1):
        super(CustomSirenModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, interm_channels, kernel_size=3, stride=1, padding=1)
        self.siren1 = SirenLayer(interm_channels, interm_channels, is_first=True)
        self.siren2 = SirenLayer(interm_channels, interm_channels)
        self.final_conv = nn.Conv2d(interm_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.siren1(x)
        x = self.siren2(x)
        x = self.final_conv(x)
        return x

# Define a memory-efficient SIREN Layer
class SirenLayer(nn.Module):
    def __init__(self, in_channels, out_channels, omega=30.0, is_first=False):
        super(SirenLayer, self).__init__()
        self.in_channels = in_channels
        self.is_first = is_first
        self.omega = omega
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.conv.weight.uniform_(-1 / self.in_channels, 1 / self.in_channels)
            else:
                self.conv.weight.uniform_(-np.sqrt(6 / self.in_channels) / self.omega, np.sqrt(6 / self.in_channels) / self.omega)

    def forward(self, x):
        return torch.sin(self.omega * self.conv(x))

# Define an optimized image loader to reduce memory usage
def image_loader(folder_path, target_size=(1000, 1000), normalize=True):
    """Load images one at a time with Rasterio, resizing to 1000x1000."""
    for file in os.listdir(folder_path):
        if file.endswith('.tif') or file.endswith('.TIF'):
            filepath = os.path.join(folder_path, file)
            print(f"Loading file: {filepath}")
            try:
                with rasterio.open(filepath) as src:
                    profile = src.profile
                    if src.width == 0 or src.height == 0:
                        print(f"Error: Image {filepath} has invalid dimensions.")
                        continue

                    image = src.read(1)
                    img_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                    # Resize to 1000x1000 for consistent output
                    img_tensor = F.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=False)

                    # Normalize to the range [0, 1] if required
                    if normalize:
                        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

                    yield img_tensor.to(device), profile
                    print(f"Loaded image shape: {img_tensor.shape}")

            except Exception as e:
                print(f"Error loading {filepath}: {e}")

# Training function with controlled output dimensions
def train(args, model, opt, scheduler):
    model.train()
    total_loss = 0

    for train_in, profile in image_loader('/content/drive/My Drive/Thesis_imagery/Ecostress/1000mpatches/Den_Haag', target_size=(1000, 1000)):
        train_tgt, _ = next(image_loader('/content/drive/My Drive/Thesis_imagery/Landsat/1000mpatches/Den_Bosch', target_size=(1000, 1000)))

        # Set both input and target to the required 1000x1000 size for training
        opt.zero_grad()
        outputs = model(train_in)
        loss = nn.MSELoss()(outputs, train_tgt)

        loss.backward()
        opt.step()
        total_loss += loss.item()

        # Clear cache and delete tensors to release memory after each iteration
        del train_in, train_tgt, outputs
        torch.cuda.empty_cache()

    scheduler.step()
    print(f"Average Loss: {total_loss:.4f}")
    return total_loss

# Define main function for data loading, training, and testing
def main(args):
    model = CustomSirenModel(in_channels=1, interm_channels=32, out_channels=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)

    # Training phase
    train_stats = train(args, model, opt, scheduler)

    # Testing phase - Save outputs
    output_dir = '/content/drive/My Drive/Thesis_imagery/Rendered_images'
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for img_index, (img_batch, profile) in enumerate(image_loader('/content/drive/My Drive/Thesis_imagery/Ecostress/1000mpatches/Den_Haag', target_size=(1000, 1000))):
            # Generate output from model
            outputs = model(img_batch)

            # Rescale outputs to [0, 255] for saving
            outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min()) * 255
            outputs = outputs.byte()

            # Save each image with a unique filename
            image_filename = f'{output_dir}/processed_image_{img_index}.tif'
            profile.update({
                'dtype': 'uint8',
                'count': 1,
                'height': 1000,
                'width': 1000,
            })

            with rasterio.open(image_filename, 'w', **profile) as dst:
                dst.write(outputs.cpu().numpy()[0, 0, :, :], 1)  # Write as a single-band image

            print(f"Processed image saved: {image_filename}")
            torch.cuda.empty_cache()

if __name__ == '__main__':
    class Args:
        device = device
        batch_size = 1
        lr = 1e-4
        num_epochs = 10

    args = Args()
    main(args)