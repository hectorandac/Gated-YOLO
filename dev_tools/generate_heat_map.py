import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from scipy.ndimage import label, binary_dilation, sum as ndi_sum

# Defining a function to parse YOLO annotations
def parse_yolo_annotations(annotation_file):
    annotations = defaultdict(list)
    
    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        cls, x, y, w, h, confidence, layer = line.strip().split()
        cls = int(cls)
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
        layer = int(layer)
        area = w * h
        confidence = float(confidence)

        annotations[cls].append((x, y, w, h, area, confidence, layer))
    
    return annotations

# Defining the heatmap function
def generate_heatmap(annotations, frame_width, frame_height, output_file):
    # Initialize heatmaps to zeros
    heatmap_small = np.zeros((frame_height, frame_width))
    heatmap_medium = np.zeros((frame_height, frame_width))
    heatmap_large = np.zeros((frame_height, frame_width))
    heatmap_xlarge = np.zeros((frame_height, frame_width))

    for cls, boxes in annotations.items():
        for x, y, w, h, area, confidence, layer in boxes:
            x = int(x * frame_width)  # Convert percentage to pixels
            y = int(y * frame_height)  # Convert percentage to pixels
            w = int(w * frame_width)  # Convert percentage to pixels
            h = int(h * frame_height)  # Convert percentage to pixels

            area = area * confidence
            
            if layer == 0:  # Small size
                heatmap_small[y:y+h, x:x+w] += area
            elif layer == 1:  # Medium size
                heatmap_medium[y:y+h, x:x+w] += area
            elif layer == 2:  # Large size
                heatmap_large[y:y+h, x:x+w] += area
            else:  # Extra-large size
                heatmap_xlarge[y:y+h, x:x+w] += area

    # Normalize each heatmap for display
    heatmaps = [heatmap_small, heatmap_medium, heatmap_large, heatmap_xlarge]
    for i in range(len(heatmaps)):
        heatmaps[i] = np.log1p(heatmaps[i])  # Logarithmic transformation
        heatmaps[i] = heatmaps[i] / np.max(heatmaps[i])  # Normalization

    # Combine the heatmaps into a single image using different colors for each size category
    combined_heatmap = np.stack([heatmaps[0], heatmaps[1], heatmaps[2]], axis=-1)

    # Save the combined heatmap as an image file
    im = Image.fromarray(np.uint8(combined_heatmap*255))
    im.save(output_file)

def generate_heatmap_2(annotations, frame_width, frame_height, output_file):
    # Initialize heatmaps to zeros
    heatmap_small = np.zeros((frame_height, frame_width))
    heatmap_medium = np.zeros((frame_height, frame_width))
    heatmap_large = np.zeros((frame_height, frame_width))
    heatmap_xlarge = np.zeros((frame_height, frame_width))

    for cls, boxes in annotations.items():
        for x, y, w, h, area, confidence, layer in boxes:
            x = int(x * frame_width)  # Convert percentage to pixels
            y = int(y * frame_height)  # Convert percentage to pixels
            w = int(w * frame_width)  # Convert percentage to pixels
            h = int(h * frame_height)  # Convert percentage to pixels

            area = area * confidence
            
            if layer == 0:  # Small size
                heatmap_small[y:y+h, x:x+w] += area
            elif layer == 1:  # Medium size
                heatmap_medium[y:y+h, x:x+w] += area
            elif layer == 2:  # Large size
                heatmap_large[y:y+h, x:x+w] += area
            else:  # Extra-large size
                heatmap_xlarge[y:y+h, x:x+w] += area

    # Normalize each heatmap for display and convert to binary
    heatmaps = [heatmap_small, heatmap_medium, heatmap_large, heatmap_xlarge]
    for i in range(len(heatmaps)):
        heatmaps[i] = np.log1p(heatmaps[i])  # Logarithmic transformation
        heatmaps[i] = heatmaps[i] / np.max(heatmaps[i])  # Normalization
        heatmaps[i] = (heatmaps[i] > 0).astype(int)  # Convert to binary

    # Dilation and connected component analysis
    for i in range(len(heatmaps)):
        heatmaps[i] = binary_dilation(heatmaps[i], iterations=10)  # Dilation to connect nearby areas
        heatmaps[i], _ = label(heatmaps[i])  # Connected component analysis to form blobs

    # Combine the heatmaps into a single image using different colors for each size category
    combined_heatmap = np.stack([heatmaps[0], heatmaps[1], heatmaps[2]], axis=-1)

    # Save the combined heatmap as an image file
    im = Image.fromarray(np.uint8(combined_heatmap*255))
    im.save(output_file)

def generate_heatmap_3(annotations, frame_width, frame_height, output_file, size_threshold=500):
    # Initialize heatmaps to zeros
    heatmap_small = np.zeros((frame_height, frame_width))
    heatmap_medium = np.zeros((frame_height, frame_width))
    heatmap_large = np.zeros((frame_height, frame_width))
    heatmap_xlarge = np.zeros((frame_height, frame_width))

    for cls, boxes in annotations.items():
        for x, y, w, h, area, confidence, layer in boxes:
            x = int(x * frame_width)  # Convert percentage to pixels
            y = int(y * frame_height)  # Convert percentage to pixels
            w = int(w * frame_width)  # Convert percentage to pixels
            h = int(h * frame_height)  # Convert percentage to pixels
            
            area = area * confidence

            if layer == 0:  # Small size
                heatmap_small[y:y+h, x:x+w] += area
            elif layer == 1:  # Medium size
                heatmap_medium[y:y+h, x:x+w] += area
            elif layer == 2:  # Large size
                heatmap_large[y:y+h, x:x+w] += area
            else:  # Extra-large size
                heatmap_xlarge[y:y+h, x:x+w] += area

    # Normalize each heatmap for display and convert to binary
    heatmaps = [heatmap_small, heatmap_medium, heatmap_large, heatmap_xlarge]
    for i in range(len(heatmaps)):
        heatmaps[i] = np.log1p(heatmaps[i])  # Logarithmic transformation
        heatmaps[i] = heatmaps[i] / np.max(heatmaps[i])  # Normalization
        heatmaps[i] = (heatmaps[i] > 0).astype(int)  # Convert to binary

    # Dilation, connected component analysis and size thresholding
    for i in range(len(heatmaps)):
        heatmaps[i] = binary_dilation(heatmaps[i], iterations=10)  # Dilation to connect nearby areas
        labeled, num_labels = label(heatmaps[i])  # Connected component analysis to form blobs
        sizes = ndi_sum(heatmaps[i], labeled, range(num_labels + 1))  # Compute blob sizes
        mask_sizes = sizes > size_threshold  # Apply size threshold
        mask_sizes[0] = 0  # Background size is set to 0
        heatmaps[i] = mask_sizes[labeled]  # Apply the mask to original image

    # Combine the heatmaps into a single image using different colors for each size category
    combined_heatmap = np.stack([heatmaps[0], heatmaps[1], heatmaps[2]], axis=-1)

    # Save the combined heatmap as an image file
    im = Image.fromarray(np.uint8(combined_heatmap*255))
    im.save(output_file)

from scipy.ndimage import label, binary_dilation, sum as ndi_sum

def generate_heatmap_4(annotations, frame_width, frame_height, output_file, size_threshold=500):
    # Initialize heatmaps to zeros
    heatmap_small = np.zeros((frame_height, frame_width))
    heatmap_medium = np.zeros((frame_height, frame_width))
    heatmap_large = np.zeros((frame_height, frame_width))
    heatmap_xlarge = np.zeros((frame_height, frame_width))

    for frame, boxes in annotations.items():
        for x, y, w, h, area, confidence, layer in boxes:
            x = int(x * frame_width)  # Convert percentage to pixels
            y = int(y * frame_height)  # Convert percentage to pixels
            w = int(w * frame_width)  # Convert percentage to pixels
            h = int(h * frame_height)  # Convert percentage to pixels
            
            area = area * confidence

            if layer == 0:  # Small size
                heatmap_small[y:y+h, x:x+w] += area
            elif layer == 1:  # Medium size
                heatmap_medium[y:y+h, x:x+w] += area
            elif layer == 2:  # Large size
                heatmap_large[y:y+h, x:x+w] += area
            else:  # Extra-large size
                heatmap_xlarge[y:y+h, x:x+w] += area

    # Normalize each heatmap for display and convert to binary
    heatmaps = [heatmap_small, heatmap_medium, heatmap_large, heatmap_xlarge]
    for i in range(len(heatmaps)):
        heatmaps[i] = np.log1p(heatmaps[i])  # Logarithmic transformation
        heatmaps[i] = heatmaps[i] / np.max(heatmaps[i])  # Normalization
        heatmaps[i] = (heatmaps[i] > 0).astype(int)  # Convert to binary

    # Dilation, connected component analysis and size thresholding
    for i in range(len(heatmaps)):
        heatmaps[i] = binary_dilation(heatmaps[i], iterations=10)  # Dilation to connect nearby areas
        labeled, num_labels = label(heatmaps[i])  # Connected component analysis to form blobs
        sizes = ndi_sum(heatmaps[i], labeled, range(num_labels + 1))  # Compute blob sizes
        mask_sizes = sizes > size_threshold  # Apply size threshold
        mask_sizes[0] = 0  # Background size is set to 0
        heatmaps[i] = mask_sizes[labeled]  # Apply the mask to original image

    # Subtract larger categories from smaller ones to prevent overlaps
    heatmaps[0] = np.logical_and(heatmaps[0], np.logical_not(heatmaps[1]))
    heatmaps[0] = np.logical_and(heatmaps[0], np.logical_not(heatmaps[2]))
    heatmaps[0] = np.logical_and(heatmaps[0], np.logical_not(heatmaps[3]))

    heatmaps[1] = np.logical_and(heatmaps[1], np.logical_not(heatmaps[2]))
    heatmaps[1] = np.logical_and(heatmaps[1], np.logical_not(heatmaps[3]))

    heatmaps[2] = np.logical_and(heatmaps[2], np.logical_not(heatmaps[3]))

    for i in range(len(heatmaps)):
        heatmaps[i] = binary_dilation(heatmaps[i], iterations=64)  # Dilation to connect nearby areas
        labeled, num_labels = label(heatmaps[i])  # Connected component analysis to form blobs
        sizes = ndi_sum(heatmaps[i], labeled, range(num_labels + 1))  # Compute blob sizes
        mask_sizes = sizes > size_threshold  # Apply size threshold
        mask_sizes[0] = 0  # Background size is set to 0
        heatmaps[i] = mask_sizes[labeled]  # Apply the mask to original image

    # Combine the heatmaps into a single image using different colors for each size category
    combined_heatmap = np.stack([heatmaps[0], heatmaps[1], heatmaps[2]], axis=-1)

    # Save the combined heatmap as an image file
    im = Image.fromarray(np.uint8(combined_heatmap*255))
    im.save(output_file)
    return heatmaps[0], heatmaps[1], heatmaps[2], heatmaps[3]

# Update with your actual frame width and height
frame_width = 1920
frame_height = 1080

annotations = parse_yolo_annotations('runs/inference/exp/labels/video1.txt')
heatmap_small, heatmap_medium, heatmap_large, heatmap_xlarge = generate_heatmap_4(annotations, frame_width, frame_height, 'heatmap4.png', 10000)

#print(heatmap_small.shape)
np.save('kernel.npy', [heatmap_small, heatmap_medium, heatmap_large, heatmap_xlarge])