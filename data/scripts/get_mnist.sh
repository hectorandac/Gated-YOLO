#!/bin/bash
# MNIST dataset
# Download command: bash get_mnist.sh
# Train command: python train.py --data mnist.yaml
# Default dataset location is next to /yolov5:
#   /parent_folder
#     /MNIST
#     /yolov5

start=$(date +%s)
mkdir -p ../tmp
cd ../tmp/

# Python script to download and process MNIST dataset
python3 - "$@" <<END
import os
import shutil
import torchvision
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

# Define directories
dataset_dir = os.path.join('..', 'MNIST')
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')
train_images_dir = os.path.join(images_dir, 'train')
val_images_dir = os.path.join(images_dir, 'val')
train_labels_dir = os.path.join(labels_dir, 'train')
val_labels_dir = os.path.join(labels_dir, 'val')

# Create directories
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Transform to convert images to tensors
transform = transforms.Compose([transforms.ToTensor()])

# Download MNIST dataset
train_dataset = datasets.MNIST(root='.', train=True, transform=transform, download=True)
val_dataset = datasets.MNIST(root='.', train=False, transform=transform, download=True)

# Helper function to save images and labels
def save_data(dataset, image_dir, label_dir, start_idx=0):
    for idx, (image, label) in enumerate(dataset):
        image_idx = idx + start_idx
        image_path = os.path.join(image_dir, f'{image_idx:05d}.png')
        label_path = os.path.join(label_dir, f'{image_idx:05d}.txt')
        
        # Convert tensor to PIL image and save
        pil_image = transforms.ToPILImage()(image)
        pil_image.save(image_path)
        
        # Convert tensor to numpy array
        np_image = np.array(pil_image)
        
        # Find bounding box for the digit
        digit_indices = np.where(np_image > 0)
        if digit_indices[0].size > 0:
            y_min, y_max = digit_indices[0].min(), digit_indices[0].max()
            x_min, x_max = digit_indices[1].min(), digit_indices[1].max()
            
            # Calculate bounding box coordinates and size
            x_center = (x_min + x_max) / 2.0 / np_image.shape[1]
            y_center = (y_min + y_max) / 2.0 / np_image.shape[0]
            width = (x_max - x_min) / np_image.shape[1]
            height = (y_max - y_min) / np_image.shape[0]
            
            # Save label in the format: class x_center y_center width height
            with open(label_path, 'w') as f:
                f.write(f'{label} {x_center} {y_center} {width} {height}\n')

# Save training data
save_data(train_dataset, train_images_dir, train_labels_dir)

# Save validation data
save_data(val_dataset, val_images_dir, val_labels_dir, start_idx=len(train_dataset))

# Create train.txt and val.txt
with open(os.path.join(dataset_dir, 'train.txt'), 'w') as f:
    for idx in range(len(train_dataset)):
        f.write(f'{os.path.join(train_images_dir, f"{idx:05d}.png")}\n')

with open(os.path.join(dataset_dir, 'val.txt'), 'w') as f:
    for idx in range(len(val_dataset)):
        f.write(f'{os.path.join(val_images_dir, f"{idx + len(train_dataset):05d}.png")}\n')

END

end=$(date +%s)
runtime=$((end - start))
echo "Completed in" $runtime "seconds"

# Clean up
rm -rf ../tmp
echo "MNIST download and preparation done."
