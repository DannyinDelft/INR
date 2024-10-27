# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:43:16 2024

@author: Gebruiker
"""

import sys
import os
import torch
import torch.nn.functional as F
import rasterio
import matplotlib.pyplot as plt
from google.colab import drive

# Step 1: Check if you are running in Google Colab
if 'google.colab' in sys.modules:
    print("You are running in Google Colab.")
else:
    print("You are not running in Google Colab. This script may not work correctly.")

# Step 2: Mount Google Drive
drive.mount('/content/drive')  # Check that your images are accessible

# Step 3: Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available.")

# Step 4: Check TensorFlow installation and GPU availability
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available for TensorFlow:", tf.config.list_physical_devices('GPU'))

# Step 5: Check Python version
print("Python version:", sys.version)

# Step 6: List installed libraries (optional for debugging)
!pip list | grep 'rasterio\|plyfile\|configargparse'

# Install any missing packages
!pip install rasterio
!pip install plyfile
!pip install configargparse

# Verify that the paths are correct by listing directories
print("Verifying directory paths:")
!ls "/content/drive/My Drive/Thesis_imagery"
!ls "/content/drive/My Drive/Thesis_imagery/Landsat/1000mpatches/Den_Haag"
!ls "/content/drive/My Drive/Thesis_imagery/Ecostress/1000mpatches/Den_Haag"

# Define the directories for Landsat and ECOSTRESS images
landsat_dir = "/content/drive/My Drive/Thesis_imagery/Landsat/1000mpatches/Den_Haag"
ecostress_dir = "/content/drive/My Drive/Thesis_imagery/Ecostress/1000mpatches/Den_Haag"

# Check if directories exist
if os.path.exists(landsat_dir):
    print("Landsat directory exists.")
    landsat_files = os.listdir(landsat_dir)
    print(f"Number of files in Landsat directory: {len(landsat_files)}")
else:
    print("Landsat directory does not exist.")

if os.path.exists(ecostress_dir):
    print("Ecostress directory exists.")
    ecostress_files = os.listdir(ecostress_dir)
    print(f"Number of files in Ecostress directory: {len(ecostress_files)}")
else:
    print("Ecostress directory does not exist.")

# Define a function to load and resize .tif images
def load_tif_images_from_folder(folder_path, target_size=None):
    """Loads .tif images from the specified folder using rasterio and resizes them if target_size is specified."""
    images = []
    print(f"Checking directory: {folder_path}")
    for file in os.listdir(folder_path):
        if file.endswith('.tif') or file.endswith('.TIF'):
            filepath = os.path.join(folder_path, file)
            print(f"Loading file: {filepath}")
            try:
                with rasterio.open(filepath) as src:
                    image = src.read(1)  # Reading the first band

                    # Convert the image to a tensor
                    img_tensor = torch.tensor(image, dtype=torch.float32)

                    # Resize the image if target_size is provided
                    if target_size is not None:
                        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                        img_tensor = F.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=False)
                        img_tensor = img_tensor.squeeze(0).squeeze(0)  # Remove the batch and channel dimensions

                    images.append(img_tensor)
                    print(f"Loaded image shape: {img_tensor.shape}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    return images

# Define the paths to your images in Google Drive
landsat_path = "/content/drive/My Drive/Thesis_imagery/Landsat/1000mpatches/Den_Haag"
ecostress_path = "/content/drive/My Drive/Thesis_imagery/Ecostress/1000mpatches/Den_Haag"

# Load the Landsat and Ecostress images, resizing them to 34x34
target_size = (34, 34)  # Resizing all images to a common size
landsat_images = load_tif_images_from_folder(landsat_path, target_size)
ecostress_images = load_tif_images_from_folder(ecostress_path, target_size)

# Your loaded images are now ready for further processing.

# Directory where processed images will be saved in Google Drive
output_dir = '/content/drive/My Drive/Thesis_imagery/Processed_images'

# Ensure the directory exists (create it if it doesn't)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Example placeholder for processed images (assuming you process them elsewhere)
processed_images = landsat_images  # Just an example, replace with actual processed images

# Loop over your processed images and save them with unique filenames
for i, processed_image in enumerate(processed_images):  # Use the correct variable name here
    output_image_path = os.path.join(output_dir, f'output_image_{i}.png')  # Unique filename for each image
    plt.imsave(output_image_path, processed_image, cmap='gray')  # Save the image
    print(f"Processed image {i} saved to: {output_image_path}")

print("Script finished successfully.")
