"""
D-FINE-Mask Criterion: Loss functions for mask prediction in D-FINE instance segmentation.
Copyright (c) 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from ...core import register
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .dfine_criterion import DFINECriterion
from .box_ops import box_cxcywh_to_xyxy


@register()
class DFINEMaskCriterion(DFINECriterion):
    """
    This class computes the loss for D-FINE-Mask.
    It extends the DFINECriterion by adding mask losses.
    """
    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        mask_weight=1.0,
        dice_weight=1.0,
        alpha=0.25,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
    ):
        """
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight
            losses: list of all the losses to be applied. See get_loss for list of available losses
            mask_weight: relative weight of the binary mask loss
            dice_weight: relative weight of the dice loss
            num_classes: number of object categories, omitting the special no-object category
            reg_max: Max number of the discrete bins in D-FINE
            boxes_weight_format: format for boxes weight
        """
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            reg_max=reg_max,
            boxes_weight_format=boxes_weight_format,
            share_matched_indices=share_matched_indices
        )
        
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the mask prediction loss.
        
        Args:
            outputs: dict of tensors with the model predictions
            targets: list of dicts containing ground truth
            indices: list of tuples with matched indices (predictions, targets)
            num_boxes: scalar for normalization
            
        Returns:
            dict of mask losses
        """
        assert "pred_masks" in outputs, "pred_masks not found in outputs"
        
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        
        # Get predicted masks for the matched predictions
        src_masks = outputs["pred_masks"][src_idx]
        
        # Get GT masks for the matched targets
        target_masks = []
        valid_targets = []
        
        for batch_idx, (_, target_indices) in enumerate(indices):
            if len(target_indices) == 0:
                continue
                
            if "masks" in targets[batch_idx]:
                target_masks.append(targets[batch_idx]["masks"][target_indices])
                valid_targets.append(batch_idx)
        
        if len(target_masks) == 0:
            # No valid targets with masks, return zero loss
            losses = {
                "loss_mask": src_masks.sum() * 0,
                "loss_dice": src_masks.sum() * 0
            }
            return losses
            
        target_masks = torch.cat(target_masks, dim=0)
        
        # Resize target masks to match predicted mask size
        if src_masks.shape[-2:] != target_masks.shape[-2:]:
            target_masks = F.interpolate(
                target_masks.unsqueeze(0), 
                size=src_masks.shape[-2:],
                mode="nearest"
            ).squeeze(0)
            
        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(
            src_masks, target_masks, reduction="none"
        )
        bce_loss = bce_loss.mean(dim=[1, 2])
        loss_mask = bce_loss.sum() / num_boxes
        
        # Dice loss
        dice_loss = self.dice_loss(
            torch.sigmoid(src_masks), target_masks, num_boxes
        )
        
        losses = {
            "loss_mask": loss_mask * self.mask_weight,
            "loss_dice": dice_loss * self.dice_weight
        }
        
        return losses
        
    def dice_loss(self, inputs, targets, num_boxes):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_boxes
        
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """
        Add mask loss to the parent get_loss method
        """
        loss_map = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "local": self.loss_local,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
        
    def forward(self, outputs, targets, **kwargs):
        """
        Override forward to include mask losses
        """
        # First compute the standard DFINE losses
        losses = super().forward(outputs, targets, **kwargs)
        
        # For aux outputs, also compute mask losses if present
        if "aux_outputs" in outputs and "pred_masks" in outputs["aux_outputs"][0]:
            for i, aux_output in enumerate(outputs["aux_outputs"]):
                # Get indices for this auxiliary output
                indices = self.matcher(aux_output, targets)["indices"]
                
                # Compute number of boxes for normalization
                num_boxes = sum(len(t["labels"]) for t in targets)
                num_boxes = torch.as_tensor(
                    [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
                )
                if is_dist_available_and_initialized():
                    torch.distributed.all_reduce(num_boxes)
                num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
                
                # Add mask losses
                l_dict = self.loss_masks(aux_output, targets, indices, num_boxes)
                l_dict = {
                    k + f"_aux_{i}": v * self.weight_dict[k.replace(f"_aux_{i}", "")]
                    for k, v in l_dict.items()
                }
                losses.update(l_dict)
                
        return losses
