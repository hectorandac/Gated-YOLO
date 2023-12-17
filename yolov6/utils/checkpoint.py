#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import shutil
import torch
import os.path as osp
from yolov6.utils.events import LOGGER
from yolov6.utils.torch_utils import fuse_model

def load_state_dict(weights, model, map_location=None):
    """Load weights from checkpoint file, only assign weights those layers' name and shape are match."""
    ckpt = torch.load(weights, map_location=map_location)
    state_dict = ckpt['model'].float().state_dict()
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    model.load_state_dict(state_dict, strict=False)
    del ckpt, state_dict, model_state_dict
    return model


def load_state_dict2(weights, model, map_location=None):
    """Load weights from checkpoint file, only assign weights those layers' name and shape are match."""
    decision_gates = torch.load('./runs/inference/exp51/gates.pt')
    ckpt = torch.load(weights, map_location=map_location)
    state_dict = ckpt['model'].float().state_dict()
    model_state_dict = model.state_dict()
    new_state_dict = {}

    for i, gate in enumerate(decision_gates):
        layer_name = ['backbone.stem', 'backbone.ERBlock_2', 'backbone.ERBlock_3', 'backbone.ERBlock_4', 'backbone.ERBlock_5', "H@J", "A@", "!@W", "A@W", "!Q@"][i]
        inactive_filters = gate.eq(0).nonzero(as_tuple=False)[:, 1]
        print(f"Gate: {layer_name} -> max index: {inactive_filters.max()}")

    for dict_layer in state_dict:
        layer_info = f"Layer: {dict_layer}"
        loaded_shape_info = f"Loaded shape: {state_dict[dict_layer].shape}"
        model_shape_info = f"Model shape: {model_state_dict[dict_layer].shape}"

        print(f"{layer_info:80} {loaded_shape_info:60} {model_shape_info:40}")


    # Iterate through each gating decision and adjust weights
    for i, gate in enumerate(decision_gates):
        layer_name = ['backbone.stem', 'backbone.ERBlock_2', 'backbone.ERBlock_3', 'backbone.ERBlock_4', 'backbone.ERBlock_5', "H@J", "A@", "!@W", "A@W", "!Q@"][i]
        inactive_filters = gate.eq(0).nonzero(as_tuple=False)[:, 1]
        print("#############")
        print(i)
        print(layer_name)
        print(inactive_filters.max())
        for dict_layer in state_dict:
            if layer_name in dict_layer:
                if state_dict[dict_layer].ndim == 4:
                    print("---------------------")
                    print(dict_layer)
                    print(state_dict[dict_layer].shape)
                    new_state_dict[dict_layer] = torch.index_select(state_dict[dict_layer], 0, inactive_filters)
                elif state_dict[dict_layer].ndim == 1:
                    new_state_dict[dict_layer] = torch.index_select(state_dict[dict_layer], 0, inactive_filters)
                else:
                    new_state_dict[dict_layer] = state_dict[dict_layer]
            else:
               new_state_dict[dict_layer] = state_dict[dict_layer]
            

    
    # Collect keys where the shape doesn't match
    shape_mismatch_keys = []

    # Filter out unmatched keys and shapes
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                new_state_dict[k] = v
            else:
                shape_mismatch_keys.append(k)

    # Print the keys with shape mismatch
    # if shape_mismatch_keys:
        #print("Shape mismatches found for the following keys:")
        # for key in shape_mismatch_keys:
            #print(f"Layer: {key}, loaded shape: {state_dict[key].shape}, model shape: {model_state_dict[key].shape}")

    model.load_state_dict(new_state_dict, strict=False)
    del ckpt, state_dict, model_state_dict
    return model

def load_checkpoint(weights, map_location=None, inplace=True, fuse=True):
    """Load model from checkpoint file."""
    LOGGER.info("Loading checkpoint from {}".format(weights))
    ckpt = torch.load(weights, map_location=map_location)  # load
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    if fuse:
        LOGGER.info("\nFusing model...")
        model = fuse_model(model).eval()
    else:
        model = model.eval()
    return model


def save_checkpoint(ckpt, is_best, save_dir, model_name=""):
    """ Save checkpoint to the disk."""
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    filename = osp.join(save_dir, model_name + '.pt')
    torch.save(ckpt, filename)
    if is_best:
        best_filename = osp.join(save_dir, 'best_ckpt.pt')
        shutil.copyfile(filename, best_filename)


def strip_optimizer(ckpt_dir, epoch):
    """Delete optimizer from saved checkpoint file"""
    for s in ['best', 'last']:
        ckpt_path = osp.join(ckpt_dir, '{}_ckpt.pt'.format(s))
        if not osp.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        if ckpt.get('ema'):
            ckpt['model'] = ckpt['ema']  # replace model with ema
        for k in ['optimizer', 'ema', 'updates']:  # keys
            ckpt[k] = None
        ckpt['epoch'] = epoch
        ckpt['model'].half()  # to FP16
        for p in ckpt['model'].parameters():
            p.requires_grad = False
        torch.save(ckpt, ckpt_path)
