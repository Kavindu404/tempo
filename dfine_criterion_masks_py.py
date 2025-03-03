"""
D-FINE Instance Segmentation: Extended Criterion Implementation
Copyright (c) 2024
"""

import torch
import torch.nn.functional as F

# This file contains the mask loss functions that should be added to
# src/zoo/dfine/dfine_criterion.py

def loss_masks(self, outputs, targets, indices, num_boxes):
    """Compute the losses related to the masks: the focal loss and dice loss.
    
    Args:
        outputs (dict): Dict of model outputs including 'pred_masks'
        targets (list): List of target dicts with 'masks'
        indices (list): List of tuples with src/tgt matching indices  
        num_boxes (int): Normalization factor
        
    Returns:
        dict: Dictionary of loss terms and values
    """
    assert "pred_masks" in outputs
    
    src_idx = self._get_src_permutation_idx(indices)
    tgt_idx = self._get_tgt_permutation_idx(indices)
    
    # Extract predicted masks for matched queries [n_matched, num_classes, mask_h, mask_w]
    src_masks = outputs["pred_masks"][src_idx]
    
    # Gather target masks from all batch elements
    target_masks = torch.cat([t["masks"][i] for t, (_, i) in zip(targets, indices)], dim=0)
    
    # Target masks: [n_matched, H, W] -> prepare for comparison with pred masks
    # Resize target masks to match predicted mask size
    h, w = src_masks.shape[-2:]
    
    # Handle different target mask shapes (some implementations store masks differently)
    if len(target_masks.shape) == 2:
        # If target_masks is flattened, reshape it
        n_masks = len(target_masks)
        target_masks = target_masks.reshape(n_masks, h, w)
    elif target_masks.shape[1:] != (h, w):
        # If dimensions don't match, resize
        target_masks = F.interpolate(
            target_masks.unsqueeze(1).float(), 
            size=(h, w), 
            mode="nearest"
        ).squeeze(1).bool()
    
    # Get target classes for selecting the appropriate mask channels
    target_classes = torch.cat([t["labels"][i] for t, (_, i) in zip(targets, indices)])
    batch_size, n_queries = outputs["pred_masks"].shape[:2]
    num_classes = outputs["pred_masks"].shape[2]
    
    # Select masks corresponding to target classes
    # For each prediction, we have [num_classes] mask channels
    # We need to select the mask corresponding to the target class
    
    # Define an index tensor for selecting the right mask
    batch_idx = src_idx[0]
    query_idx = src_idx[1]
    class_idx = target_classes
    
    # Extract masks for target classes [n_matched, mask_h, mask_w]
    # This is equivalent to src_masks[range(len(src_masks)), target_classes]
    class_masks = src_masks[range(len(src_masks)), class_idx]
    
    # Compute binary cross entropy loss (focal variant)
    alpha = 0.25
    gamma = 2.0
    
    # Binary Focal Loss for masks
    target_masks = target_masks.float()
    p = torch.sigmoid(class_masks)
    ce_loss = F.binary_cross_entropy_with_logits(class_masks, target_masks, reduction="none")
    p_t = p * target_masks + (1 - p) * (1 - target_masks)
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    if alpha >= 0:
        alpha_t = alpha * target_masks + (1 - alpha) * (1 - target_masks)
        loss = alpha_t * loss
        
    loss_mask_focal = loss.mean(1).sum() / num_boxes
    
    # Dice loss for masks
    p = torch.sigmoid(class_masks)
    numerator = 2 * (p * target_masks).sum((-2, -1))
    denominator = p.sum((-2, -1)) + target_masks.sum((-2, -1)) + 1e-6
    loss_mask_dice = 1 - (numerator / denominator).mean()
    
    return {
        "loss_mask_focal": loss_mask_focal,
        "loss_mask_dice": loss_mask_dice,
    }
