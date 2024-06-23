import os
import glob
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

def load_annotations(annotation_path):
    """ Load YOLO annotation files and parse them with a progress bar. """
    annotations = []
    files = glob.glob(os.path.join(annotation_path, '*.txt'))
    for file in tqdm(files, desc="Loading annotations"):
        with open(file, 'r') as f:
            classes = [int(line.split()[0]) for line in f.readlines()]
            annotations.append(classes)
    return annotations

def build_cooccurrence_matrix(annotations, num_classes):
    """ Build a co-occurrence matrix from annotations with progress tracking. """
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for classes in tqdm(annotations, desc="Building co-occurrence matrix"):
        for i in classes:
            for j in classes:
                matrix[i, j] += 1
    return matrix

def cluster_classes(matrix, num_groups):
    """ Cluster classes based on the co-occurrence matrix using tqdm in clustering. """
    normalized_matrix = matrix / np.max(matrix)
    clustering = AgglomerativeClustering(n_clusters=num_groups, metric='precomputed', linkage='complete')
    labels = clustering.fit_predict(1 - normalized_matrix)  # Use 1 - normalized_matrix as distance
    return labels

def main():
    annotation_path = './data/VOC/labels/train'
    num_classes = 20
    num_groups = 5

    annotations = load_annotations(annotation_path)
    matrix = build_cooccurrence_matrix(annotations, num_classes)
    class_groups = cluster_classes(matrix, num_groups)

    # Group classes based on clustering results
    groups = [[] for _ in range(num_groups)]
    for class_id, group_id in enumerate(class_groups):
        groups[group_id].append(class_id)

    print("Class groups based on co-occurrence:")
    for idx, group in enumerate(groups):
        print(f"Group {idx + 1}: {group}")

if __name__ == "__main__":
    main()