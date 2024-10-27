# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:43:16 2024

@author: Gebruiker
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.cuda.amp as amp
import numpy as np
import os
import matplotlib.pyplot as plt
import rasterio
from tqdm import tqdm
import time
import skimage
from modules.modeling_utils import get_model, construct_model_args, get_summary_dict
from modules.utils import get_psnr, network_spec

# Ensure Google Drive is mounted
from google.colab import drive
drive.mount('/content/drive')

print("The script train_2d_images_attempt_20_09_2024_2043.py has started.")

print(torch.cuda.is_available())  # This should return True if CUDA is available

def load_images_without_resampling(folder_path):
    """Loads images from the specified folder without resampling."""
    images = []
    print(f"Checking directory: {folder_path}")
    
    for file in os.listdir(folder_path):
        if file.endswith('.TIF') or file.endswith('.tif'):
            filepath = os.path.join(folder_path, file)
            print(f"Loading file: {filepath}")
            try:
                with rasterio.open(filepath) as src:
                    image = src.read(1)  # Read the first band
                    img_tensor = torch.tensor(image, dtype=torch.float32)  # Convert to tensor
                    
                    images.append(img_tensor)
                    print(f"Loaded image shape: {img_tensor.shape}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    return images

def load_and_resample_ecostress_images(folder_path, target_resolution=(30, 30), min_output_size=(34, 34)):
    """Loads and resamples ECOSTRESS images to match the Landsat resolution (30x30 meters)."""
    images = []
    print(f"Checking directory: {folder_path}")
    
    for file in os.listdir(folder_path):
        if file.endswith('.TIF') or file.endswith('.tif'):
            filepath = os.path.join(folder_path, file)
            print(f"Loading file: {filepath}")
            try:
                with rasterio.open(filepath) as src:
                    current_resolution = src.res  # Get the current resolution (meters/pixel)
                    image = src.read(1)  # Read the first band
                    height, width = image.shape

                    # Calculate new dimensions based on target resolution (30m)
                    new_width = int(width * (current_resolution[0] / target_resolution[0]))  # New width in pixels
                    new_height = int(height * (current_resolution[1] / target_resolution[1]))  # New height in pixels

                    # Ensure the resampled dimensions are at least the minimum size
                    new_width = max(new_width, min_output_size[1])
                    new_height = max(new_height, min_output_size[0])

                    print(f"Resampling {filepath} to size: {new_height}x{new_width}")

                    # Convert image to tensor
                    img_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                    # Resample image to match target resolution
                    img_resampled = F.interpolate(img_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
                    img_resampled = img_resampled.squeeze(0).squeeze(0)  # Remove added dimensions
                    
                    images.append(img_resampled)
                    print(f"Loaded and resampled image shape: {img_resampled.shape}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    return images

def preprocess(img_tensor):
    return img_tensor * 2 - 1

def postprocess(img_tensor):
    return torch.clamp(((img_tensor + 1) / 2), 0, 1).squeeze(-1).detach().cpu().numpy()

def hw2batch(hw_tensor):
    if len(hw_tensor.shape) == 3:  # (b, h, w)
        hw_tensor = hw_tensor.unsqueeze(1)  # Shape becomes (b, 1, h, w)
    b, c, h, w = hw_tensor.shape
    return hw_tensor, h, w

def batch2hw(batched_tensor, h, w):
    c = batched_tensor.size(1)
    hw = batched_tensor.view(-1, h, w, c).contiguous()
    return hw

def train(args, train_data, model, opt, scheduler):
    train_in, train_tgt = train_data
    train_in = preprocess(train_in).to(args.device).clone().detach()
    train_tgt = preprocess(train_tgt).to(args.device).clone().detach()

    train_batched_input, train_h, train_w = hw2batch(train_in)
    model.train()
    total_loss = 0

    scaler = amp.GradScaler()

    for i in range(len(train_batched_input)):
        opt.zero_grad()

        with amp.autocast():
            outputs = model(train_batched_input[i])
            predicted_output = outputs['output']
            loss_function = nn.MSELoss()

            train_tgt_resized = F.interpolate(train_tgt[i].unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

            loss = loss_function(predicted_output, train_tgt_resized)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item()

    scheduler.step()

    avg_loss = total_loss / len(train_batched_input)

    print(f"Loss: {avg_loss:.4f}")
    return avg_loss

def render(model, render_input, device='cuda'):
    batched_input, h, w = hw2batch(render_input)
    z_init = torch.zeros(1, model.interm_channels).to(device)
    rec = postprocess(batch2hw(model(batched_input, z_init)['output'], h, w))
    return rec

def main(args):
    # Load Landsat images without resampling
    landsat_images = load_images_without_resampling('/content/drive/My Drive/Thesis_imagery/Landsat/1000mpatches/Den_Haag')

    # Load and resample ECOSTRESS images to match 30x30 meter resolution
    ecostress_images = load_and_resample_ecostress_images('/content/drive/My Drive/Thesis_imagery/Ecostress/1000mpatches/Den_Haag', target_resolution=(30, 30))

    if not landsat_images:
        print("Error: No Landsat images were loaded. Please check the directory.")
        return
    if not ecostress_images:
        print("Error: No ECOSTRESS images were loaded. Please check the directory.")
        return

    full_x = np.linspace(0, 1, 512) * 2 - 1
    full_x_grid = torch.tensor(np.stack(np.meshgrid(full_x, full_x), axis=-1)[None, :, :], dtype=torch.float32).to(args.device)

    x_train = full_x_grid
    y_train = torch.stack(landsat_images).to(args.device)  # Stack resized images into a tensor

    model_args = {
        'model_type': 'implicit',
        'in_channels': 7109,
        'interm_channels': 10704,
        'out_channels': 1,
        'input_scale': 256,
        'n_layers': 4
    }

    model = get_model(model_args).to(args.device)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)

    train_stats = train(args, (x_train, y_train), model, opt, scheduler)
    
    with torch.no_grad():
        for i, img in enumerate(ecostress_images):
            input_tensor = img.unsqueeze(0).to(args.device)
            rendered_img = render(model, input_tensor, device=args.device)

            output_dir = '/content/drive/My Drive/Thesis_imagery/Rendered_images'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            try:
                image_filename = f'{output_dir}/ecostress_rendered_{i}.png'
                plt.imsave(image_filename, rendered_img[0], cmap='gray')
                print(f"Rendered ECOSTRESS image saved as: {image_filename}")
            except Exception as e:
                print(f"Error saving image {i}: {e}")

print("The script train_2d_images_attempt_20_09_2024_2043.py has finished successfully.")

if __name__ == '__main__':
    class Args:
        device = 'cuda'
        batch_size = 16
        lr = 1e-4
        verbose = False
        no_skip_solver = False
        epoch = 1
        num_epochs = 10

    args = Args()
    main(args)
