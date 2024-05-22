#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
import logging
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import csv

def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)
NCOLS = min(200, shutil.get_terminal_size().columns)


def load_yaml(file_path):
    """Load data from yaml file."""
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    return data_dict


def save_yaml(data_dict, save_path):
    """Save data to yaml file"""
    with open(save_path, 'w') as f:
        yaml.safe_dump(data_dict, f, sort_keys=False)

def calculate_open_gates_percentage(gates):
    # Assuming gates is a 1D tensor of binary values (0 or 1)
    num_open_gates = gates[0].sum()
    total_gates = gates[0].numel()  # Total number of gates in the tensor
    percentage_open = (num_open_gates / total_gates) * 100
    return percentage_open.item()  # Convert to a Python number if necessary

def data_to_image(data):
    """Converts the matplotlib plot specified by 'figure' to a NumPy image array."""
    # Attach the figure to a canvas and draw it
    canvas = FigureCanvas(data)
    canvas.draw()

    # Convert to a NumPy array
    image = np.array(canvas.buffer_rgba())

    # Close the figure to free memory
    plt.close(data)

    return image

def save_proportions_to_file(proportions, filename='gate_proportions.csv'):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(proportions)

def write_tblog(tblogger, epoch, results, lrs, losses, closed=None):
    """Display mAP and loss information to log."""
    tblogger.add_scalar("val/mAP@0.5", results[0], epoch + 1)
    tblogger.add_scalar("val/mAP@0.50:0.95", results[1], epoch + 1)

    tblogger.add_scalar("train/iou_loss", losses[0], epoch + 1)
    tblogger.add_scalar("train/dist_focalloss", losses[1], epoch + 1)
    tblogger.add_scalar("train/cls_loss", losses[2], epoch + 1)

    if len(losses) >= 4:
        tblogger.add_scalar("train/gtg_loss", losses[3], epoch + 1)
        tblogger.add_scalar("train/gtg_closed", closed, epoch + 1)

    tblogger.add_scalar("x/lr0", lrs[0], epoch + 1)
    tblogger.add_scalar("x/lr1", lrs[1], epoch + 1)
    tblogger.add_scalar("x/lr2", lrs[2], epoch + 1)


def write_tbimg(tblogger, imgs, step, type='train'):
    """Display train_batch and validation predictions to tensorboard."""
    if type == 'train':
        tblogger.add_image(f'train_batch', imgs, step + 1, dataformats='HWC')
    elif type == 'val':
        for idx, img in enumerate(imgs):
            tblogger.add_image(f'val_img_{idx + 1}', img, step + 1, dataformats='HWC')
    else:
        LOGGER.warning('WARNING: Unknown image type to visualize.\n')
