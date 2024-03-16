#!/usr/bin/env python3

import numpy as np
import cv2
import os
import argparse

def apply_mask(img_path, mask_path):
    # Load the image
    img = cv2.imread(img_path)
    
    # Ensure the image is in the range [0, 1]
    img_normalized = img.astype(np.float32) / 255.0

    # Load the mask from the .npy file
    masks = np.load(mask_path)
    
    # Ensure the dimensions match
    if img.shape[0] != masks.shape[1] or img.shape[1] != masks.shape[2]:
        print(img.shape)
        print(masks.shape)
        print("Error: Dimensions of the image and the mask do not match!")
        return
    
    output_dir = "output_masks_list"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Apply the mask for each head/layer
    for idx, mask in enumerate(masks):
        masked_img = img_normalized * mask[:, :, np.newaxis]  # The mask is applied to each channel
        masked_img = (masked_img * 255).astype(np.uint8)  # Convert back to the range [0, 255]
        
        # Save the masked image
        output_path = os.path.join(output_dir, f"masked_img_layer_{idx}.jpg")
        cv2.imwrite(output_path, masked_img)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply masks to an image based on a .npy file.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('mask_path', type=str, help='Path to the .npy mask file')
    args = parser.parse_args()
    apply_mask(args.image_path, args.mask_path)
