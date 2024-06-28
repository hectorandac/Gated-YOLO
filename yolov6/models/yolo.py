#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
from yolov6.layers.common import *
from yolov6.utils.torch_utils import initialize_weights
from yolov6.models.efficientrep import *
from yolov6.models.reppan import *
from yolov6.utils.events import LOGGER
from yolov6.models.gaternet import GaterNetwork
import itertools

start_bold = "\033[1m"
end_bold = "\033[0m"

class Model(nn.Module):
    export = False
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, config, channels=3, num_classes=None, fuse_ab=False, distill_ns=False, enable_gater_net=False, fixed_gates_enabled=False):  # model, input channels, number of classes
        super().__init__()
        # Build network
        num_layers = config.model.head.num_layers
        self.backbone, self.neck, self.detect, self.gater_b, self.gater_n, self.gater_h = build_network(config, channels, num_classes, num_layers, fuse_ab=fuse_ab, distill_ns=distill_ns, enable_gater_net=enable_gater_net, fixed_gates_enable=fixed_gates_enabled)
        self.enable_gater_net = enable_gater_net
        self.fixed_gates_enable = fixed_gates_enabled

        self.backbone.enable_gater_net = enable_gater_net
        self.neck.enable_gater_net = enable_gater_net

        # Init Detect head
        self.stride = self.detect.stride
        self.detect.initialize_biases()

        # Init weights
        initialize_weights(self)

    def forward(self, x):
        export_mode = torch.onnx.is_in_onnx_export() or self.export
        closed_gates_percentage = 0

        gating_decisions_a = None
        gating_decisions_b = None
        gating_decisions_c = None

        f_out = None
        
        if self.enable_gater_net and not self.fixed_gates_enable:
            gating_decisions_a, closed_gates_percentage_a, f_out = self.gater_b(x, training=self.training, epsilon=1.0 if self.training else 0)
        x = self.backbone(x, gating_decisions_a)

        if self.enable_gater_net and not self.fixed_gates_enable:
            gating_decisions_b, closed_gates_percentage_b, _f_out = self.gater_n(x, training=self.training, epsilon=1.0 if self.training else 0)
        x = self.neck(x, gating_decisions_b)

        if self.enable_gater_net and not self.fixed_gates_enable:
            gating_decisions_c, closed_gates_percentage_c, _f_out = self.gater_h(x, training=self.training, epsilon=1.0 if self.training else 0)

        if not export_mode:
            featmaps = []
            featmaps.extend(x)
        if self.training:
            x = self.detect(x, gating_decisions_c)
            gating_decisions = [*gating_decisions_a, *gating_decisions_b, *gating_decisions_c]
            closed_gates_percentage = (closed_gates_percentage_a + closed_gates_percentage_b + closed_gates_percentage_c) / 3
            return x, gating_decisions, featmaps, closed_gates_percentage, f_out
        else:
            x = self.detect(x, gating_decisions_c)
            gating_decisions = [*gating_decisions_a, *gating_decisions_b, *gating_decisions_c]
            return x, gating_decisions, None

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

def build_network(config, channels, num_classes, num_layers, fuse_ab=False, distill_ns=False, enable_gater_net=False, fixed_gates_enable=False):
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

    if enable_gater_net and not fixed_gates_enable:
        # Extract state dictionaries
        backbone_dict = backbone.state_dict()
        neck_dict = neck.state_dict()
        head_dict = head.state_dict()

        sorted_head_layers = sorted(head_dict.keys(), key=sort_key)
        sorted_head_dict = {layer: head_dict[layer] for layer in sorted_head_layers}
        head_dict = sorted_head_dict

        # Define gate-able layers
        includes = ("conv.weight", "rbr_1x1.bn.weight", "upsample_transpose.weight")
        excludes = ("rbr_dense", "rbr_1x1.conv.weight", "proj_conv.weight")
        ignore_gates_for = [
            "stems.0.block.conv.weight", "cls_convs.0.block.conv.weight", "reg_convs.0.block.conv.weight",
            "stems.1.block.conv.weight", "cls_convs.1.block.conv.weight", "reg_convs.1.block.conv.weight",
            "stems.2.block.conv.weight", "cls_convs.2.block.conv.weight", "reg_convs.2.block.conv.weight",
            "proj_conv.weight"
        ]

        # Function to calculate gateable layers and sections
        def calculate_sections(state_dict, includes, excludes, ignore_gates_for):
            sections = []
            for layer_name, param in state_dict.items():
                if layer_name in ignore_gates_for:
                    continue
                if any(include in layer_name for include in includes) and not any(exclude in layer_name for exclude in excludes):
                    num_gates = param.shape[0]
                    sections.append(num_gates)
                    print(f"✅ GATING: {layer_name:<50} Channels -> {num_gates}")
            return sections

        # Calculate gate sections for each subnetwork
        backbone_sections = calculate_sections(backbone_dict, includes, excludes, ignore_gates_for)
        neck_sections = calculate_sections(neck_dict, includes, excludes, ignore_gates_for)
        head_sections = calculate_sections(head_dict, includes, excludes, [])

        # Print gate-able layer's count and total gates
        print(f"\nBackbone Gate-able Layer's count {len(backbone_sections)} with total of {sum(backbone_sections)} Gates\n")
        print(f"Neck Gate-able Layer's count {len(neck_sections)} with total of {sum(neck_sections)} Gates\n")
        print(f"Head Gate-able Layer's count {len(head_sections)} with total of {sum(head_sections)} Gates\n")

        # Calculate cumulative gates channels
        cumulative_backbone_gates = list(itertools.accumulate(backbone_sections))
        cumulative_neck_gates = list(itertools.accumulate(neck_sections))
        cumulative_head_gates = list(itertools.accumulate(head_sections))

        input_channels_list = [32, 64, 128]
        feature_extractors = [
            GaterNetwork.dimensionality_reduction(input_channels, bottleneck_size=256) 
            for input_channels in input_channels_list
        ]

        # Define GaterNetwork for each sub-network
        gater_backbone = GaterNetwork(
            feature_extractor_arch=GaterNetwork.create_feature_extractor_darknet53,
            num_features=1024,
            bottleneck_size=256,
            num_filters=sum(backbone_sections),
            sections=cumulative_backbone_gates,
        )

        gater_neck = GaterNetwork(
            feature_extractor_arch=None,
            feature_extractors=feature_extractors,
            num_features=256 * len(input_channels_list),
            bottleneck_size=256,
            num_filters=sum(neck_sections),
            sections=cumulative_neck_gates,
        )

        gater_head = GaterNetwork(
            feature_extractor_arch=None,
            feature_extractors=feature_extractors,
            num_features=256 * len(input_channels_list),
            bottleneck_size=256,
            num_filters=sum(head_sections),
            sections=cumulative_head_gates,
        )

        return backbone, neck, head, gater_backbone, gater_neck, gater_head

    return backbone, neck, head, None, None, None


def build_model(cfg, num_classes, device, fuse_ab=False, distill_ns=False, enable_gater_net=False):
    model = Model(cfg, channels=3, num_classes=num_classes, fuse_ab=fuse_ab, distill_ns=distill_ns, enable_gater_net=enable_gater_net).to(device)
    return model
