import os
import random
import shutil
from pathlib import Path

def extract_classes(label_dir):
    class_names = set()
    for filename in os.listdir(label_dir):
        with open(Path(label_dir, filename), 'r') as file:
            for line in file:
                class_name = line.split()[0]
                class_names.add(class_name)
    return sorted(list(class_names))

def kitti_to_yolo(kitti_annotation, img_width, img_height, class_mapping):
    parts = kitti_annotation.split()
    class_id = class_mapping.get(parts[0])

    if class_id is None:
        return None

    bbox = [float(parts[i]) for i in range(4, 8)]
    x_center = ((bbox[0] + bbox[2]) / 2) / img_width
    y_center = ((bbox[1] + bbox[3]) / 2) / img_height
    width = (bbox[2] - bbox[0]) / img_width
    height = (bbox[3] - bbox[1]) / img_height

    return f'{class_id} {x_center} {y_center} {width} {height}'

def process_dataset(root_dir, output_dir, split_ratio=0.7):
    # Extract classes from the dataset
    class_names = extract_classes(Path(root_dir, 'training', 'label_2'))
    class_mapping = {name: i for i, name in enumerate(class_names)}

    # Create directories
    for part in ["images", "labels"]:
        for subset in ["train", "val", "test"]:
            Path(output_dir, part, subset).mkdir(parents=True, exist_ok=True)

    all_files = os.listdir(Path(root_dir, 'training', 'label_2'))
    random.shuffle(all_files)
    split_index = int(len(all_files) * split_ratio)

    for subset, files in [('train', all_files[:split_index]), ('val', all_files[split_index:])]:
        for filename in files:
            base_filename = filename.split('.')[0]
            img_src = Path(root_dir, 'training', 'image_2', base_filename + '.png')
            label_src = Path(root_dir, 'training', 'label_2', base_filename + '.txt')

            with open(label_src, 'r') as file, open(Path(output_dir, 'labels', subset, base_filename + '.txt'), 'w') as out_file:
                for line in file:
                    yolo_line = kitti_to_yolo(line, 1242, 375, class_mapping)
                    if yolo_line:
                        out_file.write(yolo_line + '\n')

            shutil.copy(img_src, Path(output_dir, 'images', subset, base_filename + '.png'))

    # Copy test data
    for filename in os.listdir(Path(root_dir, 'testing', 'image_2')):
        shutil.copy(Path(root_dir, 'testing', 'image_2', filename), Path(output_dir, 'images', 'test', filename))

    return class_names

# Example usage
root_dir = '/home/hectorandac/Downloads/test'
output_dir = '/home/hectorandac/Documents/yolo-v6-size-invariant/YOLOv6/dataset/dataset_2'
class_names = process_dataset(root_dir, output_dir)

print("Extracted class names:", class_names)
