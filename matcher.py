import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou

class HungarianMatcher(nn.Module):
    def __init__(self, cls_cost=2.0, bbox_cost=5.0, giou_cost=2.0, mask_cost=2.0, dice_cost=5.0):
        super().__init__()
        self.cls_cost = cls_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        self.mask_cost = mask_cost
        self.dice_cost = dice_cost

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs:
          - pred_logits: (B, Q, C+1)
          - pred_boxes: (B, Q, 4) in [0,1] cxcywh
          - pred_masks: (B, Q, H, W) logits
        targets: list of dict with 'labels','boxes'(xyxy in image space, but provided as 0..W/H; we will expect normalized cxcywh),
                 'masks' (0/1 HxW)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)  # (B,Q,C+1)
        out_bbox = outputs["pred_boxes"]               # (B,Q,4) cxcywh in [0,1]
        out_masks = outputs["pred_masks"].sigmoid()    # (B,Q,H,W)

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]  # xyxy pixel; we need normalized cxcywh
            H, W = targets[b]["masks"].shape[-2:]
            # normalize
            xyxy = tgt_bbox.clone()
            xyxy[:, 0::2] /= W
            xyxy[:, 1::2] /= H
            # convert xyxy -> cxcywh
            cxcywh = torch.zeros_like(xyxy)
            cxcywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
            cxcywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
            cxcywh[:, 2] = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=1e-6)
            cxcywh[:, 3] = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=1e-6)

            # classification cost: prob of target class (class id 1), use negative log-prob
            # single foreground class; index 1 corresponds to foreground; 0 is background? We'll set C=1, so foreground=0, last class is no-object.
            # For safety, compute using provided labels-1 but clamp to 0 since single class.
            C = out_prob.shape[-1] - 1  # number of foreground classes = 1
            fg_class_idx = torch.zeros_like(tgt_ids)  # all zeros
            cost_cls = -out_prob[b, :, fg_class_idx].squeeze(-1)  # (Q, T)

            # bbox costs
            cost_bbox = torch.cdist(out_bbox[b], cxcywh, p=1)  # (Q,T)
            # giou on xyxy: convert out_bbox to xyxy
            ob = out_bbox[b]
            ob_xyxy = torch.zeros_like(ob)
            ob_xyxy[:, 0] = ob[:, 0] - ob[:, 2] / 2
            ob_xyxy[:, 1] = ob[:, 1] - ob[:, 3] / 2
            ob_xyxy[:, 2] = ob[:, 0] + ob[:, 2] / 2
            ob_xyxy[:, 3] = ob[:, 1] + ob[:, 3] / 2
            # scale to (0,1) like cxcywh (tgt in (0,1) already)
            giou = generalized_box_iou(ob_xyxy, xyxy)
            cost_giou = -giou

            # mask costs: focal-like (BCE) and dice
            # downsample masks for cheaper matching
            ph, pw = out_masks.shape[-2] // 4, out_masks.shape[-1] // 4
            pm = torch.nn.functional.interpolate(out_masks[b].unsqueeze(1), size=(ph,pw), mode="bilinear", align_corners=False).squeeze(1)
            tm = torch.nn.functional.interpolate(targets[b]["masks"].float().unsqueeze(1), size=(ph,pw), mode="nearest").squeeze(1)
            pm = pm.flatten(1)     # (Q, ph*pw)
            tm = tm.flatten(1).T   # (T, ph*pw)
            # focal part: -|p - y|
            cost_mask = torch.cdist(pm, tm, p=1) / (ph*pw)

            # dice cost
            pm_sig = pm
            tm_sig = tm
            numerator = 2 * (pm_sig @ tm_sig.T)
            denom = (pm_sig.sum(dim=1)[:, None] + tm_sig.sum(dim=1)[None, :]).clamp(min=1e-6)
            dice = 1 - (numerator + 1e-6) / (denom + 1e-6)
            cost_dice = dice

            C_total = (self.cls_cost * cost_cls
                       + self.bbox_cost * cost_bbox
                       + self.giou_cost * cost_giou
                       + self.mask_cost * cost_mask
                       + self.dice_cost * cost_dice)

            C_total = C_total.cpu()
            q_idx, t_idx = linear_sum_assignment(C_total)
            indices.append((torch.as_tensor(q_idx, dtype=torch.int64),
                            torch.as_tensor(t_idx, dtype=torch.int64)))
        return indices
