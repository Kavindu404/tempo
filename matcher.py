import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple
import numpy as np

def box_cxcywh_to_xyxy(x):
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format"""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format"""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    iou, union = box_iou(boxes1, boxes2)
    
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]
    
    return iou - (area - union) / area

def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def box_area(boxes):
    """Compute the area of a set of bounding boxes"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def dice_loss(inputs, targets, smooth=1.0):
    """Compute dice loss"""
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    
    intersection = (inputs * targets).sum(1)
    dice = (2. * intersection + smooth) / (inputs.sum(1) + targets.sum(1) + smooth)
    return 1 - dice

def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """Compute sigmoid focal loss"""
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss

class HungarianMatcher(torch.nn.Module):
    """Hungarian matcher for instance segmentation.
    
    This class computes an assignment between targets and predictions of the model.
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as no-object).
    """
    
    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, cost_bbox: float = 1):
        """Creates the matcher
        
        Params:
            cost_class: relative weight of the classification error in the matching cost
            cost_mask: relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: relative weight of the dice loss of the binary mask in the matching cost  
            cost_bbox: relative weight of the L1 error of the bounding box coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_bbox = cost_bbox
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0 or cost_bbox != 0, "all costs cant be 0"
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching
        
        Params:
            outputs: dict containing:
                - "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                - "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks
                - "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                
            targets: list of targets (len(targets) = batch_size), where each target is a dict containing:
                - "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                - "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks
                - "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_mask = outputs["pred_masks"].flatten(0, 1)  # [batch_size * num_queries, H_pred, W_pred]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        # Also concat the target labels and masks
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_mask = torch.cat([v["masks"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # Convert target boxes to [cx, cy, w, h] format and normalize
        if len(tgt_bbox) > 0:
            # Convert from [x, y, w, h] to [cx, cy, w, h] and normalize
            tgt_bbox_norm = tgt_bbox.clone()
            tgt_bbox_norm[:, 0] = (tgt_bbox[:, 0] + tgt_bbox[:, 2] / 2) / targets[0]["orig_size"][1]  # cx
            tgt_bbox_norm[:, 1] = (tgt_bbox[:, 1] + tgt_bbox[:, 3] / 2) / targets[0]["orig_size"][0]  # cy  
            tgt_bbox_norm[:, 2] = tgt_bbox[:, 2] / targets[0]["orig_size"][1]  # w
            tgt_bbox_norm[:, 3] = tgt_bbox[:, 3] / targets[0]["orig_size"][0]  # h
        else:
            tgt_bbox_norm = tgt_bbox
        
        # Compute the classification cost
        if len(tgt_ids) > 0:
            # Since we have single class, targets are all class 1, predictions are [no_object, class_1]
            cost_class = -out_prob[:, tgt_ids]  # Use negative probability as cost
        else:
            cost_class = torch.zeros(out_prob.shape[0], 0, device=out_prob.device)
        
        # Compute the mask costs
        if len(tgt_mask) > 0:
            # Resize prediction masks to match target size
            H_pred, W_pred = out_mask.shape[-2:]
            H_tgt, W_tgt = tgt_mask.shape[-2:]
            
            if H_pred != H_tgt or W_pred != W_tgt:
                # Interpolate predicted masks to target size
                out_mask_resized = F.interpolate(
                    out_mask.unsqueeze(1), 
                    size=(H_tgt, W_tgt), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
            else:
                out_mask_resized = out_mask
            
            # Flatten for cost computation
            out_mask_flat = out_mask_resized.flatten(1)  # [num_queries, H*W]
            tgt_mask_flat = tgt_mask.flatten(1)  # [num_targets, H*W]
            
            # Focal loss cost
            cost_mask = sigmoid_focal_loss(
                out_mask_flat[:, None, :], 
                tgt_mask_flat[None, :, :].float()
            ).mean(-1)  # [num_queries, num_targets]
            
            # Dice loss cost  
            cost_dice = dice_loss(
                out_mask_flat[:, None, :], 
                tgt_mask_flat[None, :, :].float()
            )  # [num_queries, num_targets]
        else:
            cost_mask = torch.zeros(out_prob.shape[0], 0, device=out_prob.device)
            cost_dice = torch.zeros(out_prob.shape[0], 0, device=out_prob.device)
        
        # Compute the bbox cost
        if len(tgt_bbox_norm) > 0:
            cost_bbox = torch.cdist(out_bbox, tgt_bbox_norm, p=1)  # L1 cost
        else:
            cost_bbox = torch.zeros(out_prob.shape[0], 0, device=out_prob.device)
        
        # Final cost matrix
        C = self.cost_class * cost_class + self.cost_mask * cost_mask + \
            self.cost_dice * cost_dice + self.cost_bbox * cost_bbox
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def get_world_size():
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
