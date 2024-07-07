#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox, bbox2dist, xywh2xyxy, box_iou
from yolov6.utils.figure_iou import IOUloss
from yolov6.assigners.atss_assigner import ATSSAssigner
from yolov6.assigners.tal_assigner import TaskAlignedAssigner
from collections import Counter

class_to_group = {
    9: 0, 11: 0, 12: 0, 16: 0,  # Group 1
    0: 1, 2: 1, 3: 1,           # Group 2
    18: 2,                      # Group 3
    4: 3, 7: 3, 8: 3, 10: 3, 14: 3, 15: 3, 17: 3, 19: 3,  # Group 4
    1: 4, 5: 4, 6: 4, 13: 4     # Group 5
}

# Define the new loss functions
def annealing_sparsity_loss(gate_values, current_sparsity):
    sparsity = torch.mean(gate_values)
    # print("######################", current_sparsity, sparsity)
    return F.mse_loss(sparsity, torch.tensor(current_sparsity, dtype=sparsity.dtype, device=sparsity.device))

def soft_threshold(gate_values, target_sparsity):
    threshold = torch.quantile(gate_values, target_sparsity)
    return (gate_values > threshold).float() * gate_values
class ComputeLoss:
    '''Loss computation func.'''
    def __init__(self,
                 fpn_strides=[8, 16, 32],
                 grid_cell_size=5.0,
                 grid_cell_offset=0.5,
                 num_classes=80,
                 ori_img_size=640,
                 warmup_epoch=4,
                 use_dfl=True,
                 reg_max=16,
                 iou_type='giou',
                 loss_weight={
                     'class': 1.8,
                     'iou': 3.5,
                     'dfl': 0.5,
                     'gtg': 0.5,
                     'gtg_decay': 0.25
                    },
                 ):

        self.fpn_strides = fpn_strides
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = num_classes
        self.ori_img_size = ori_img_size

        self.warmup_epoch = warmup_epoch
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max, self.use_dfl, self.iou_type).cuda()
        self.loss_weight = loss_weight

    def __call__(
        self,
        outputs,
        targets,        
        epoch_num,
        step_num,
        gating_decision,
        f_out
    ):
        feats, pred_scores, pred_distri = outputs
        
        anchors, anchor_points, n_anchors_list, stride_tensor = \
               generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device)

        assert pred_scores.type() == pred_distri.type()
        gt_bboxes_scale = torch.full((1,4), self.ori_img_size).type_as(pred_scores)
        batch_size = pred_scores.shape[0]

        # targets
        targets =self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:] #xyxy
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pboxes
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri) #xyxy

        try:
            if epoch_num < self.warmup_epoch:
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.warmup_assigner(
                        anchors,
                        n_anchors_list,
                        gt_labels,
                        gt_bboxes,
                        mask_gt,
                        pred_bboxes.detach() * stride_tensor)
            else:
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.formal_assigner(
                        pred_scores.detach(),
                        pred_bboxes.detach() * stride_tensor,
                        anchor_points,
                        gt_labels,
                        gt_bboxes,
                        mask_gt)

        except RuntimeError:
            print(
                "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                    CPU mode is applied in this batch. If you want to avoid this issue, \
                    try to reduce the batch size or image size."
            )
            torch.cuda.empty_cache()
            print("------------CPU Mode for This Batch-------------")
            if epoch_num < self.warmup_epoch:
                _anchors = anchors.cpu().float()
                _n_anchors_list = n_anchors_list
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _stride_tensor = stride_tensor.cpu().float()

                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.warmup_assigner(
                        _anchors,
                        _n_anchors_list,
                        _gt_labels,
                        _gt_bboxes,
                        _mask_gt,
                        _pred_bboxes * _stride_tensor)

            else:
                _pred_scores = pred_scores.detach().cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _anchor_points = anchor_points.cpu().float()
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _stride_tensor = stride_tensor.cpu().float()

                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.formal_assigner(
                        _pred_scores,
                        _pred_bboxes * _stride_tensor,
                        _anchor_points,
                        _gt_labels,
                        _gt_bboxes,
                        _mask_gt)

            target_labels = target_labels.cuda()
            target_bboxes = target_bboxes.cuda()
            target_scores = target_scores.cuda()
            fg_mask = fg_mask.cuda()
        #Dynamic release GPU memory
        if step_num % 10 == 0:
            torch.cuda.empty_cache()

        # rescale bbox
        target_bboxes /= stride_tensor

        # cls loss
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)

        target_scores_sum = target_scores.sum()
		# avoid devide zero error, devide by zero will cause loss to be inf or nan.
        # if target_scores_sum is 0, loss_cls equals to 0 alson
        if target_scores_sum > 1:
            loss_cls /= target_scores_sum

        # bbox loss
        loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes,
                                            target_scores, target_scores_sum, fg_mask)

        # Compute the total loss with adaptive weights
        loss = self.loss_weight['class'] * loss_cls + \
            self.loss_weight['iou'] * loss_iou + \
            self.loss_weight['dfl'] * loss_dfl

        # Prepare the loss components for return
        loss_components = [
            (self.loss_weight['iou'] * loss_iou).unsqueeze(0),
            (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0),
            (self.loss_weight['class'] * loss_cls).unsqueeze(0)
        ]

        # Compute gating loss (L1 regularization term)
        # Compute the gating loss
        gating_loss = None
        if gating_decision is not None:
            g_reconstructed = torch.cat([gd[0].flatten(start_dim=1) for gd in gating_decision], dim=1)

            #print(g_reconstructed.shape)
            #print(g_reconstructed[0][0:10])
            
            # Replace L1 based loss with the new annealing sparsity loss

            target_loss = 1 - 0.4
            gating_loss = annealing_sparsity_loss(g_reconstructed, target_loss)
            loss += gating_loss
            loss_components.append(gating_loss.unsqueeze(0))
            
            # Enable soft gating towards the end of training (last 10% of epochs)
            if epoch_num > 400 * 0.9:
                g_reconstructed.data = soft_threshold(g_reconstructed.data, 0.4)

            if f_out is not None:
                subgroup_idx = self.determine_dominant_subgroup(target_labels)
                subgroup_loss_fn = nn.CrossEntropyLoss()
                subgroup_loss = subgroup_loss_fn(f_out, subgroup_idx)
                loss += subgroup_loss
                loss_components.append(subgroup_loss.unsqueeze(0))

        return loss, torch.cat(loss_components).detach(), target_loss
    
    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = torch.from_numpy(np.array(list(map(lambda l:l + [[-1,0,0,0,0]]*(max_len - len(l)), targets_list)))[:,1:,:]).to(targets.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:] = xywh2xyxy(batch_target)
        return targets

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(self.proj.to(pred_dist.device))
        return dist2bbox(pred_dist, anchor_points)
    
    def map_classes_to_subgroups(self, gt_labels):
        # Map class labels to subgroup labels
        subgroup_labels = torch.zeros_like(gt_labels)
        for cls, grp in class_to_group.items():
            subgroup_labels[gt_labels == cls] = grp
        return subgroup_labels

    def determine_dominant_subgroup(self, gt_labels):
        # Determine the dominant subgroup for each image in the batch
        batch_size = gt_labels.shape[0]
        dominant_subgroups = torch.zeros(batch_size, dtype=torch.long, device=gt_labels.device)
        for i in range(batch_size):
            valid_labels = gt_labels[i][gt_labels[i] != self.num_classes]
            if valid_labels.numel() == 0:
                dominant_subgroups[i] = 0
            else:
                subgroup_labels = self.map_classes_to_subgroups(valid_labels)
                # Determine the most common subgroup
                dominant_subgroups[i] = subgroup_labels.mode().values.item()
        return dominant_subgroups

class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score,gt_score, label, alpha=0.75, gamma=2.0):

        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()

        return loss


class BboxLoss(nn.Module):
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format='xyxy', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):

        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                  bbox_mask).reshape([-1, 4])
            target_bboxes_pos = torch.masked_select(
                target_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     target_bboxes_pos) * bbox_weight
            if target_scores_sum > 1:
                loss_iou = loss_iou.sum() / target_scores_sum
            else:
                loss_iou = loss_iou.sum()

            # dfl loss
            if self.use_dfl:
                dist_mask = fg_mask.unsqueeze(-1).repeat(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                target_ltrb_pos = torch.masked_select(
                    target_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                        target_ltrb_pos) * bbox_weight
                if target_scores_sum > 1:
                    loss_dfl = loss_dfl.sum() / target_scores_sum
                else:
                    loss_dfl = loss_dfl.sum()
            else:
                loss_dfl = pred_dist.sum() * 0.

        else:
            loss_iou = pred_dist.sum() * 0.
            loss_dfl = pred_dist.sum() * 0.

        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)
