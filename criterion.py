import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from matcher import HungarianMatcher, sigmoid_focal_loss, dice_loss, box_cxcywh_to_xyxy, generalized_box_iou

class SetCriterion(nn.Module):
    """Compute the losses for Mask2Former-style model.
    
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    
    def __init__(self, config, matcher, weight_dict, eos_coef, losses):
        """Create the criterion.
        
        Parameters:
            config: configuration object
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied
        """
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        # For classification loss
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef  # no-object class weight
        self.register_buffer('empty_weight', empty_weight)
        
    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # [bs, num_queries, num_classes+1]
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                   dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, H_W]
        """
        assert "pred_masks" in outputs
        
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]  # [bs, num_queries, H, W]
        src_masks = src_masks[src_idx]  # [num_matched, H, W]
        
        # Concatenate all target masks
        target_masks = torch.cat([t['masks'] for t in targets], dim=0)  # [total_targets, H_tgt, W_tgt]
        target_masks = target_masks[tgt_idx]  # [num_matched, H_tgt, W_tgt]
        
        if src_masks.shape[0] == 0:
            # No matched predictions
            losses = {
                "loss_mask": torch.tensor(0.0, device=src_masks.device, requires_grad=True),
                "loss_dice": torch.tensor(0.0, device=src_masks.device, requires_grad=True),
            }
            return losses
        
        # Resize predictions to match target size
        H_pred, W_pred = src_masks.shape[-2:]
        H_tgt, W_tgt = target_masks.shape[-2:]
        
        if H_pred != H_tgt or W_pred != W_tgt:
            src_masks = F.interpolate(
                src_masks.unsqueeze(1), 
                size=(H_tgt, W_tgt), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
        
        # Flatten spatial dimensions
        src_masks = src_masks.flatten(1)  # [num_matched, H*W]
        target_masks = target_masks.flatten(1).float()  # [num_matched, H*W]
        
        # For matched pairs, inputs and targets should have the same shape
        # No need for broadcasting here since we have matched pairs
        
        # Focal loss - use simple BCE focal loss for matched pairs
        prob = src_masks.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction="none")
        p_t = prob * target_masks + (1 - prob) * (1 - target_masks)
        focal_loss = ce_loss * ((1 - p_t) ** 2.0)  # gamma=2.0
        alpha = 0.25
        alpha_t = alpha * target_masks + (1 - alpha) * (1 - target_masks)
        loss_mask = (alpha_t * focal_loss).mean()
        
        # Dice loss - simple dice for matched pairs
        intersection = (src_masks.sigmoid() * target_masks).sum(1)
        dice = (2. * intersection + 1.0) / (src_masks.sigmoid().sum(1) + target_masks.sum(1) + 1.0)
        loss_dice = (1 - dice).mean()
        
        losses = {
            "loss_mask": loss_mask,
            "loss_dice": loss_dice,
        }
        return losses
    
    def loss_boxes(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (x, y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]  # [num_matched, 4]
        
        # Get target boxes and normalize them
        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)  # [total_targets, 4]
        tgt_idx = self._get_tgt_permutation_idx(indices)
        target_boxes = target_boxes[tgt_idx]  # [num_matched, 4]
        
        if src_boxes.shape[0] == 0:
            # No matched predictions
            losses = {
                "loss_bbox": torch.tensor(0.0, device=src_boxes.device, requires_grad=True),
                "loss_giou": torch.tensor(0.0, device=src_boxes.device, requires_grad=True),
            }
            return losses
        
        # Normalize target boxes to [0, 1] format (cx, cy, w, h)
        # Assuming target boxes are in [x, y, w, h] format
        img_h, img_w = targets[0]["orig_size"]
        target_boxes_norm = target_boxes.clone()
        target_boxes_norm[:, 0] = (target_boxes[:, 0] + target_boxes[:, 2] / 2) / img_w  # cx
        target_boxes_norm[:, 1] = (target_boxes[:, 1] + target_boxes[:, 3] / 2) / img_h  # cy
        target_boxes_norm[:, 2] = target_boxes[:, 2] / img_w  # w
        target_boxes_norm[:, 3] = target_boxes[:, 3] / img_h  # h
        
        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes_norm, reduction='mean')
        
        # GIoU loss
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes_norm)
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy))
        loss_giou = loss_giou.mean()
        
        losses = {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks, 
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_masks, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / torch.distributed.get_world_size() if torch.distributed.is_initialized() else num_masks, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        return losses

def build_criterion(config):
    """Build the criterion for the model"""
    
    # Create matcher
    matcher = HungarianMatcher(
        cost_class=config.cls_loss_weight,
        cost_mask=config.mask_loss_weight, 
        cost_dice=config.dice_loss_weight,
        cost_bbox=config.bbox_loss_weight,
    )
    
    # Weight dictionary for different losses
    weight_dict = {
        'loss_ce': config.cls_loss_weight,
        'loss_mask': config.mask_loss_weight,
        'loss_dice': config.dice_loss_weight,
        'loss_bbox': config.bbox_loss_weight,
        'loss_giou': config.giou_loss_weight,
    }
    
    # List of losses to compute
    losses = ['labels', 'masks', 'boxes']
    
    # EOS coefficient (weight for no-object class)
    eos_coef = 0.1
    
    criterion = SetCriterion(
        config=config,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=eos_coef,
        losses=losses
    )
    
    return criterion, weight_dict
