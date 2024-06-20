#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import cv2
import time
import math
import torch
import numpy as np
import os.path as osp

from tqdm import tqdm
from pathlib import Path
from PIL import ImageFont
from collections import deque

from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend, CounterA
from yolov6.data.data_augment import letterbox
from yolov6.data.datasets import LoadData
from yolov6.utils.nms import non_max_suppression

# from fvcore.nn import FlopCountAnalysis

from scipy.ndimage import label, binary_dilation, sum as ndi_sum
from PIL import Image
from collections import defaultdict

class Inferer:
    def __init__(self, source, webcam, webcam_addr, weights, device, yaml, img_size, half, fixed_gates, enable_fixed_gates):

        self.__dict__.update(locals())

        # Init model
        self.device = device
        self.img_size = img_size
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')
        self.model = DetectBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.class_names = load_yaml(yaml)['names']
        self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check image size
        self.half = half

        # Switch model to deploy status
        self.model_switch(self.model.model, self.img_size)

        if enable_fixed_gates:
            self.model.model.fixed_gates_enable = enable_fixed_gates
            self.prune_weights(self.model.model, torch.load(fixed_gates))

        # Half precision
        if self.half & (self.device.type != 'cpu'):
            self.model.model.half()
        else:
            self.model.model.float()
            self.half = False

        # Load data
        self.webcam = webcam
        self.webcam_addr = webcam_addr
        self.files = LoadData(source, webcam, webcam_addr)
        self.source = source

    def prune_weights(self, model, gates):
        LOGGER.info("Pruning model for deployment")
        model.fixed_gates = gates

        ignore_gates_for = []

        # Includes now allow both weights and biases
        includes = (".weight", ".bias")
        excludes = ("gater.", "detect.")  # Removed ".bias" from excludes to process biases

        gate_index = 0

        for layer_name, param in model.named_parameters():
            if any(include in layer_name for include in includes) and not any(exclude in layer_name for exclude in excludes) and layer_name not in ignore_gates_for:
                if gate_index < len(gates):
                    LOGGER.info(f"Processing layer: {layer_name}")
                    gate, size = gates[gate_index]
                    zeroed_channels = 0

                    if size != None:
                        desired_shape = torch.Size([1, size[1], 1, 1])
                        gate = torch.zeros(desired_shape, dtype=torch.bool)

                    if ".weight" in layer_name:
                        for i, keep in enumerate(gate.squeeze()):
                            if not keep.item():
                                param.data[i].zero_()
                                if param.grad is not None:
                                    param.grad.data[i].zero_()
                                zeroed_channels += 1
                        LOGGER.info(f"Zeroed {zeroed_channels}/{gate.numel()} channels in weight: {layer_name}")

                    elif ".bias" in layer_name:
                        for i, keep in enumerate(gate.squeeze()):
                            if not keep.item():
                                param.data[i].zero_()
                                if param.grad is not None:
                                    param.grad.data[i].zero_()
                        LOGGER.info(f"Zeroed bias in layer: {layer_name}")
                        gate_index += 1

                    # Print the actual weights or biases after zeroing out unnecessary channels
                    flattened_param = param.data.cpu().numpy()
                    LOGGER.info(f"Parameters after pruning (flattened): {flattened_param.shape}")
                else:
                    LOGGER.warning(f"No gate found for layer: {layer_name}, skipping pruning for this layer")


    def model_switch(self, model, img_size):
        ''' Model switch to deploy status '''
        from yolov6.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
                layer.recompute_scale_factor = None  # torch 1.11.0 compatibility

        LOGGER.info("Switch model to deploy modality.")

    def infer(
        self, conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, save_img, hide_labels, hide_conf, view_img=True,
        analyze=False, enable_gater_net=False):
        ''' Model Inference and results visualization '''
        
        vid_path, vid_writer, windows = None, None, []
        fps_calculator = CalcFPS()
        min_fps = float('inf')
        max_fps = 0
        avg_fps = 0

        if analyze and enable_gater_net:
            gating_accumulator = None
            base_dim = []

        for img_src, img_path, vid_cap in tqdm(self.files):
            CounterA.reset()
            img, img_src = self.process_image(img_src, self.img_size, self.stride, self.half)
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img[None]
                # expand for batch dim
            t1 = time.time()
            pred_results, gating_decision = self.model(img)
            #CounterA.reset()
            #lops = FlopCountAnalysis(self.model, img)
            #print(f"Total FLOPS: {flops.total()}")
            det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
            t2 = time.time()

            if analyze and enable_gater_net:
                if gating_accumulator is None:
                    # Initialize gating_accumulator with the same shape as gating_decision, except for the batch dimension
                    gating_accumulator = [torch.zeros_like(gd[0].sum(dim=0, keepdim=True)) for gd in gating_decision]
                    base_dim = [gd[1] for gd in gating_decision]
                
                for i, gd in enumerate(gating_decision):
                    # Sum gd over batch dimension while keeping the dimension
                    gating_accumulator[i] += gd[0].sum(dim=0, keepdim=True)

            if self.webcam:
                save_path = osp.join(save_dir, self.webcam_addr)
                txt_path = osp.join(save_dir, self.webcam_addr)
            else:
                # Create output files in nested dirs that mirrors the structure of the images' dirs
                rel_path = osp.relpath(osp.dirname(img_path), osp.dirname(self.source))
                save_path = osp.join(save_dir, rel_path, osp.basename(img_path))  # im.jpg
                txt_path = osp.join(save_dir, rel_path, 'labels', osp.splitext(osp.basename(img_path))[0])
                os.makedirs(osp.join(save_dir, rel_path), exist_ok=True)

            gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            img_ori = img_src.copy()

            # check image and font
            assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
            self.font_check()

            if len(det):
                det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (self.box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:
                        class_num = int(cls)  # integer class
                        label = None if hide_labels else (self.class_names[class_num] if hide_conf else f'{self.class_names[class_num]} {conf:.2f}')

                        self.plot_box_and_label(img_ori, 2, xyxy, label, color=self.generate_colors(class_num, True))

                img_src = np.asarray(img_ori)

            # FPS counter
            fps_calculator.update(1.0 / (t2 - t1))
            avg_fps = fps_calculator.accumulate()

            current_fps = 1.0 / (t2 - t1)
            min_fps = min(min_fps, current_fps)
            max_fps = max(max_fps, current_fps)

            if self.files.type == 'video':
                self.draw_text(
                    img_src,
                    f"FPS: {avg_fps:0.1f}",
                    pos=(20, 20),
                    font_scale=1.0,
                    text_color=(204, 85, 17),
                    text_color_bg=(255, 255, 255),
                    font_thickness=2,
                )

            if view_img:
                if img_path not in windows:
                    windows.append(img_path)
                    cv2.namedWindow(str(img_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(img_path), img_src.shape[1], img_src.shape[0])
                cv2.imshow(str(img_path), img_src)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if self.files.type == 'image':
                    cv2.imwrite(save_path, img_src)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, img_ori.shape[1], img_ori.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(img_src)
        
        print(f"Average FPS: {avg_fps}")
        print(f"Minimum FPS: {min_fps}")
        print(f"Maximum FPS: {max_fps}")
        # print(prof.display(show_events=False))

        if analyze:
            print("Analyzing detections")
            self.files = LoadData(self.source, self.webcam, self.webcam_addr)
            img_src, img_path, vid_cap = next(iter(self.files))
            if self.files.type == 'video':
                ret, frame = vid_cap.read()
                if not ret:
                    print(f"Can't read the frame from {img_path}")
                else:
                    x_image = frame
            elif self.files.type == 'image':
                x = img_src

            if enable_gater_net:
                print("Getting gating_frequencies...")
                stored_gates = []
                
                for i, accumulator in enumerate(gating_accumulator):
                    # Normalize by the number of samples to get the frequency for this section
                    gating_frequency = accumulator.squeeze() / 19120  # TODO: Ensure accumulator is correctly squeezed

                    # Identify gates below the threshold for this section
                    completely_off_gates = (gating_frequency == 0)
                    always_on_gates = (gating_frequency > 0)

                    print(f"Section {i}:")
                    print(f"  Percentage of filters that are completely off: {completely_off_gates.float().mean() * 100:.2f}%")
                    print(f"  Percentage of filters that are always active: {always_on_gates.float().mean() * 100:.2f}%")

                    if completely_off_gates.float().mean() > 0.99 and base_dim[i] is not None:
                        stored_gates.append([None, base_dim[i]])
                    else:
                        gating_decision_for_section = ~completely_off_gates.unsqueeze(0)
                        stored_gates.append([gating_decision_for_section.unsqueeze(-1).unsqueeze(-1), None])

                for section in stored_gates:
                    gate = ""
                    if section[0] is not None:
                        gate = section[0].shape
                    print(F"Gate: {gate}, Tensor dimension: {section[1]}")

                print("Storing gates in to gates.pt")
                torch.save(stored_gates, f"{save_dir}/gates.pt")


            
            # annotations = self.parse_yolo_annotations(txt_path + '.txt')
            # heat_maps = self.generate_heatmap(annotations, x_image.shape[1], x_image.shape[0], save_dir + '/heatmap.png', 64000)
            
            img, img_src = self.process_image(img_src, self.img_size, self.stride, self.half)
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img[None]

    def first_and_others(generator):
        iterator = iter(generator)
        first_item = next(iterator)
        yield first_item
        yield first_item
        yield from iterator 

    @staticmethod
    def parse_yolo_annotations(annotation_file):
        annotations = defaultdict(list)
    
        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            cls, x, y, w, h, confidence = line.strip().split()
            cls = int(cls)
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            confidence = float(confidence)

            annotations[cls].append((x, y, w, h, confidence))
        
        return annotations


    @staticmethod
    def generate_heatmap(annotations, frame_width, frame_height, output_file, size_threshold=500):
        # Initialize heatmaps to zeros
        heatmap_small = np.zeros((frame_height, frame_width))
        heatmap_medium = np.zeros((frame_height, frame_width))
        heatmap_large = np.zeros((frame_height, frame_width))
        heatmap_xlarge = np.zeros((frame_height, frame_width))
        
        overlap_count = np.zeros((frame_height, frame_width, 4), dtype=int)

        for i in range(len(annotations)):
            for x, y, w, h, confidence, layer in tqdm(annotations[i], desc="Detections"):
                x = int(x * frame_width)  # Convert percentage to pixels
                y = int(y * frame_height)  # Convert percentage to pixels
                w = int(w * frame_width)  # Convert percentage to pixels
                h = int(h * frame_height)  # Convert percentage to pixels

                area = w * h * confidence

                if layer == 0:  # Small size
                    heatmap_small[y:y+h, x:x+w] += area
                elif layer == 1:  # Medium size
                    heatmap_medium[y:y+h, x:x+w] += area
                elif layer == 2:  # Large size
                    heatmap_large[y:y+h, x:x+w] += area
                else:  # Extra-large size
                    heatmap_xlarge[y:y+h, x:x+w] += area

                # Increase count in overlap_count
                overlap_count[y:y+h, x:x+w, layer] += 1

        # Decide precedence and Create Binary Maps
        chosen_layer = np.argmax(overlap_count * np.array([4,3,2,1]), axis=-1)
        no_detection_mask = np.all(overlap_count == 0, axis=-1)
        chosen_layer[no_detection_mask] = -1

        heatmaps = [heatmap_small, heatmap_medium, heatmap_large, heatmap_xlarge]
        for i in range(4):
            heatmaps[i] = (chosen_layer == i).astype(int)

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

    @staticmethod
    def process_image(img_src, img_size, stride, half):
        '''Process image before image inference.'''
        image = letterbox(img_src, img_size, stride=stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src
    
    @staticmethod
    def generate_head_specific_kernels(original_kernel, img_shape, cell_size, num_heads):
        """
        Generate a set of point-wise kernels based on an original kernel.

        Parameters:
            original_kernel (np.ndarray): The original kernel.
            img_shape (tuple): Shape of the original image as (height, width).
            cell_size (int): Size of each cell in the original kernel.
            num_heads (int): Number of head layers.

        Returns:
            np.ndarray: An array of new kernels for each head, starting from index 1.
        """

        # Calculate the number of cells along width and height
        num_cells_w = img_shape[1] // cell_size
        num_cells_h = img_shape[0] // cell_size

        # Initialize an array of kernels, one for each head
        new_kernels = np.zeros((num_heads-1, img_shape[0], img_shape[1]))

        # Populate the new kernels
        for h in range(1, num_heads):
            for y in range(num_cells_h):
                for x in range(num_cells_w):
                    # Find the head with the maximum detection in this cell
                    max_head = np.argmax(original_kernel[y, x])

                    # Fill the corresponding area in the new kernel
                    start_y = y * cell_size
                    end_y = start_y + cell_size
                    start_x = x * cell_size
                    end_x = start_x + cell_size

                    # If this cell's max head is 'h', set this region to 1 in new_kernels[h]
                    new_kernels[h-1, start_y:end_y, start_x:end_x] = int(max_head == h)

        return new_kernels

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    @staticmethod
    def draw_text(
        img,
        text,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        pos=(0, 0),
        font_scale=1,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0),
    ):

        offset = (5, 5)
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        rec_start = tuple(x - y for x, y in zip(pos, offset))
        rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
        cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
        cv2.putText(
            img,
            text,
            (x, int(y + text_h + font_scale - 1)),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        return text_size

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

    @staticmethod
    def font_check(font='./yolov6/utils/Arial.ttf', size=10):
        # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
        assert osp.exists(font), f'font path not exists: {font}'
        try:
            return ImageFont.truetype(str(font) if font.exists() else font.name, size)
        except Exception as e:  # download if missing
            return ImageFont.truetype(str(font), size)

    @staticmethod
    def box_convert(x):
        # Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    @staticmethod
    def generate_colors(i, bgr=False):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        palette = []
        for iter in hex:
            h = '#' + iter
            palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
        num = len(palette)
        color = palette[int(i) % num]
        return (color[2], color[1], color[0]) if bgr else color

class CalcFPS:
    def __init__(self, nsamples: int = 50):
        self.framerate = deque(maxlen=nsamples)

    def update(self, duration: float):
        self.framerate.append(duration)

    def accumulate(self):
        if len(self.framerate) > 1:
            return np.average(self.framerate)
        else:
            return 0.0
