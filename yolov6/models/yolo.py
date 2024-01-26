#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.layers.common import *
from yolov6.utils.torch_utils import initialize_weights
from yolov6.models.efficientrep import *
from yolov6.models.reppan import *
from yolov6.utils.events import LOGGER


class Model(nn.Module):
    export = False
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, config, channels=3, num_classes=None, fuse_ab=False, distill_ns=False, enable_gater_net=False):  # model, input channels, number of classes
        super().__init__()
        # Build network
        num_layers = config.model.head.num_layers
        self.backbone, self.neck, self.detect, self.gater = build_network(config, channels, num_classes, num_layers, fuse_ab=fuse_ab, distill_ns=distill_ns)
        self.detect.inference_with_mask = False
        self.neck.inference_with_mask = False
        self.enable_gater_net = enable_gater_net

        # Init Detect head
        self.stride = self.detect.stride
        self.detect.initialize_biases()

        # Init weights
        initialize_weights(self)

    def forward(self, x):
        export_mode = torch.onnx.is_in_onnx_export() or self.export
        if not self.enable_gater_net:
            x = self.backbone(x)
            x = self.neck(x)
            if not export_mode:
                featmaps = []
                featmaps.extend(x)
            x = self.detect(x)
            if self.training:
                return (*x, None), None
            else:
                return x, None
        else:
            gating_decisions = self.gater(x, training=self.training, epsilon=1.0 if self.training else 0)
            x = self.backbone(x, gating_decisions)
            x = self.neck(x, gating_decisions)
            if not export_mode:
                featmaps = []
                featmaps.extend(x)
            if self.training:
                x, cls_score_list, reg_distri_list = self.detect(x, gating_decisions)
                return (x, cls_score_list, reg_distri_list, gating_decisions), gating_decisions
            else:
                x = self.detect(x, gating_decisions)
                return x, gating_decisions
    
    def prune_regions(self, x, source_mask, path):
        CounterA.reset()
        if not self.enable_gater_net:
            x = self.backbone(x)
            x = self.neck(x)
        else:
            gating_decisions = self.gater(x, training=self.training, epsilon=1.0 if self.training else 0)
            x = self.backbone(x, gating_decisions)
            x = self.neck(x, gating_decisions)

        mask = [None for _ in range(len(x))]
        n_type = x[0].dtype
        n_device = x[0].device

        for i in range(min(len(source_mask), len(x))):
            b, g, h, w = x[i].shape
            original_mask_tensor = torch.tensor(source_mask[i], dtype=n_type, device=n_device)
            interpolated_mask = F.interpolate(original_mask_tensor.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest')
            expanded_mask = interpolated_mask.expand(b, g, -1, -1)
            
            # Check if the tensor is all zeros. If it is, set the mask entry to None.
            if not expanded_mask.eq(0).all():
                mask[i] = expanded_mask

        torch.save(mask, path + "/masks.pt")


    def _apply(self, fn):
        self = super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self


def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor

def sort_key(layer_name):
    parts = layer_name.split('.')
    # Check if the layer name can be split and has a numerical ID
    if len(parts) > 1 and parts[1].isdigit():
        return int(parts[1])
    else:
        # Return a default value that keeps the layer in its original position
        return -1

def build_network(config, channels, num_classes, num_layers, fuse_ab=False, distill_ns=False):
    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    fuse_P2 = config.model.backbone.get('fuse_P2')
    cspsppf = config.model.backbone.get('cspsppf')
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    use_dfl = config.model.head.use_dfl
    reg_max = config.model.head.reg_max
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

    block = get_block(config.training_mode)
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)

    if 'CSP' in config.model.backbone.type:

        if "stage_block_type" in config.model.backbone:
            stage_block_type = config.model.backbone.stage_block_type
        else:
            stage_block_type = "BepC3"  #default

        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.backbone.csp_e,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf,
            stage_block_type=stage_block_type
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.neck.csp_e,
            stage_block_type=stage_block_type
        )
    else:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )

    if distill_ns:
        from yolov6.models.heads.effidehead_distill_ns import Detect, build_effidehead_layer
        if num_layers != 3:
            LOGGER.error('ERROR in: Distill mode not fit on n/s models with P6 head.\n')
            exit()
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    elif fuse_ab:
        from yolov6.models.heads.effidehead_fuseab import Detect, build_effidehead_layer
        anchors_init = config.model.head.anchors_init
        head_layers = build_effidehead_layer(channels_list, 3, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, anchors_init, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    else:
        from yolov6.models.effidehead import Detect, build_effidehead_layer
        head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max, num_layers=num_layers)
        head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    sections = []
    backbone_dict = backbone.state_dict()
    neck_dict = neck.state_dict()
    head_dict = head.state_dict()

    sorted_head_layers = sorted(head_dict.keys(), key=sort_key)
    sorted_head_dict = {layer: head_dict[layer] for layer in sorted_head_layers}

    combined_dict = {**backbone_dict, **neck_dict, **sorted_head_dict}

    for layer_name in combined_dict:
        if ("conv.weight" in layer_name and "rbr_dense" not in layer_name and "rbr_1x1.conv.weight" not in layer_name) or ("rbr_1x1.bn.weight" in layer_name) or ("upsample_transpose.weight" in layer_name) or ("reg_preds" in layer_name and "weight" in layer_name) or ("cls_preds" in layer_name and "weight" in layer_name):
            num_gates = combined_dict[layer_name].shape[0]
            sections.append(num_gates)
            print(f"✅ GATING: {layer_name} with Channels -> {num_gates}")
        else:
            print(f"❌ NOT GATING: {layer_name} with Channels -> {combined_dict[layer_name].shape}")

    print(f"Gate-able Layer's count {len(sections)} with total of {sum(sections)} Gates")

    cumulativeGatesChannels = list(itertools.accumulate(sections))
    num_filters = sum(sections) if channels_list else 0

    gater = GaterNetwork(
        feature_extractor_arch=GaterNetwork.create_feature_extractor_resnet18,
        num_features=204800,         # Number of output features from the feature extractor
        num_filters=num_filters,
        sections=cumulativeGatesChannels,
        bottleneck_size=128,         # Size of the bottleneck in the GaterNetwork
    )

    return backbone, neck, head, gater


def build_model(cfg, num_classes, device, fuse_ab=False, distill_ns=False, enable_gater_net=False):
    model = Model(cfg, channels=3, num_classes=num_classes, fuse_ab=fuse_ab, distill_ns=distill_ns, enable_gater_net=enable_gater_net).to(device)
    return model
