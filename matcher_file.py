import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_mask=1, cost_dice=1, cost_bbox=1, cost_giou=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        out_mask = outputs["pred_masks"].flatten(0, 1)
        
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_mask = torch.cat([v["masks"] for v in targets])
        
        if tgt_ids.shape[0] == 0:
            return [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)) for _ in range(bs)]
        
        # Classification cost
        cost_class = -out_prob[:, 0].unsqueeze(1)
        
        # L1 cost for bbox
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # GIoU cost for bbox
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
        
        # Mask cost
        out_mask = F.interpolate(out_mask.unsqueeze(1), 
                                 size=tgt_mask.shape[-2:], 
                                 mode="bilinear", 
                                 align_corners=False).squeeze(1)
        
        out_mask = out_mask.flatten(1)
        tgt_mask = tgt_mask.flatten(1).float()
        
        cost_mask = torch.cdist(out_mask, tgt_mask, p=1) / tgt_mask.shape[-1]
        
        # Dice cost
        out_mask_sig = out_mask.sigmoid()
        numerator = 2 * torch.einsum('nc,mc->nm', out_mask_sig, tgt_mask)
        denominator = out_mask_sig.sum(-1)[:, None] + tgt_mask.sum(-1)[None, :]
        cost_dice = 1 - (numerator + 1) / (denominator + 1)
        
        # Final cost matrix
        C = (self.cost_mask * cost_mask + 
             self.cost_class * cost_class + 
             self.cost_dice * cost_dice + 
             self.cost_bbox * cost_bbox + 
             self.cost_giou * cost_giou)
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["labels"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def generalized_box_iou(boxes1, boxes2):
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)
    
    inter_mins = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_maxs = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inter_wh = (inter_maxs - inter_mins).clamp(min=0)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1[:, None] + area2 - inter_area
    
    iou = inter_area / union_area
    
    hull_mins = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    hull_maxs = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    hull_wh = hull_maxs - hull_mins
    hull_area = hull_wh[:, :, 0] * hull_wh[:, :, 1]
    
    giou = iou - (hull_area - union_area) / hull_area
    return giou