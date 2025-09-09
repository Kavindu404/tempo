import torch
from torch import nn
from torchvision.ops import generalized_box_iou

def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="none"):
    prob = inputs.sigmoid()
    ce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss

def dice_loss(inputs, targets, eps=1e-6):
    # inputs: logits (B,Q,H,W) -> sigmoid
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    num = 2 * (inputs * targets).sum(dim=1)
    den = inputs.sum(dim=1) + targets.sum(dim=1) + eps
    loss = 1 - (num + eps) / (den + eps)
    return loss

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weights, no_object_weight=0.1,
                 cls_alpha=0.25, cls_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weights = weights
        self.no_object_weight = no_object_weight
        self.cls_alpha = cls_alpha
        self.cls_gamma = cls_gamma

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = no_object_weight
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs["pred_logits"]  # (B,Q,C+1)
        B, Q, C1 = src_logits.shape
        # Build target classes (C foreground classes=1 here)
        target_classes = torch.full((B, Q), C1 - 1, dtype=torch.int64, device=src_logits.device)  # no-object
        for b, (src_idx, tgt_idx) in enumerate(indices):
            target_classes[b, src_idx] = 0  # since we have one foreground class
        # Convert to one-hot over C+1, then focal on foreground vs background
        # We'll compute CE on all classes with weighting:
        loss_ce = nn.functional.cross_entropy(src_logits.transpose(1,2), target_classes, weight=self.empty_weight)
        return {"loss_cls": loss_ce}

    def loss_boxes(self, outputs, targets, indices):
        src_boxes = outputs["pred_boxes"]
        losses = {"loss_bbox": torch.tensor(0., device=src_boxes.device),
                  "loss_giou": torch.tensor(0., device=src_boxes.device)}
        count = 0
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx)==0: 
                continue
            sb = src_boxes[b, src_idx]  # cxcywh
            tb_xyxy = targets[b]["boxes"].float().clone()
            H, W = targets[b]["masks"].shape[-2:]
            tb_xyxy[:, 0::2] /= W
            tb_xyxy[:, 1::2] /= H
            # convert to cxcywh
            tb = torch.zeros_like(tb_xyxy)
            tb[:, 0] = (tb_xyxy[:, 0] + tb_xyxy[:, 2]) / 2
            tb[:, 1] = (tb_xyxy[:, 1] + tb_xyxy[:, 3]) / 2
            tb[:, 2] = (tb_xyxy[:, 2] - tb_xyxy[:, 0]).clamp(min=1e-6)
            tb[:, 3] = (tb_xyxy[:, 3] - tb_xyxy[:, 1]).clamp(min=1e-6)

            loss_l1 = nn.functional.l1_loss(sb, tb, reduction="sum")
            # giou: need xyxy
            def cxcywh_to_xyxy(bx):
                out = torch.zeros_like(bx)
                out[:, 0] = bx[:, 0] - bx[:, 2]/2
                out[:, 1] = bx[:, 1] - bx[:, 3]/2
                out[:, 2] = bx[:, 0] + bx[:, 2]/2
                out[:, 3] = bx[:, 1] + bx[:, 3]/2
                return out
            giou = generalized_box_iou(cxcywh_to_xyxy(sb), cxcywh_to_xyxy(tb))
            loss_giou = (1 - giou).sum()

            losses["loss_bbox"] += loss_l1
            losses["loss_giou"] += loss_giou
            count += len(src_idx)

        for k in losses:
            if count > 0:
                losses[k] /= count
        return losses

    def loss_masks(self, outputs, targets, indices):
        src_masks = outputs["pred_masks"]  # (B,Q,H,W) logits
        loss_focal_sum = torch.tensor(0., device=src_masks.device)
        loss_dice_sum = torch.tensor(0., device=src_masks.device)
        count = 0
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx)==0: 
                continue
            pm = src_masks[b, src_idx]        # (S,H,W)
            tm = targets[b]["masks"][tgt_idx].float()  # (S,H,W) 0/1
            # ensure same size
            if pm.shape[-2:] != tm.shape[-2:]:
                tm = torch.nn.functional.interpolate(tm.unsqueeze(1), size=pm.shape[-2:], mode="nearest").squeeze(1)
            # focal
            lf = sigmoid_focal_loss(pm, tm, alpha=self.cls_alpha, gamma=self.cls_gamma, reduction="sum")
            # dice
            ld = dice_loss(pm, tm).sum()
            loss_focal_sum += lf
            loss_dice_sum += ld
            count += pm.shape[0]

        if count > 0:
            loss_focal_sum /= count
            loss_dice_sum /= count
        return {"loss_mask_focal": loss_focal_sum, "loss_mask_dice": loss_dice_sum}

    def forward(self, outputs, targets, indices):
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices))
        losses.update(self.loss_masks(outputs, targets, indices))

        # weighted sum
        w = self.weights
        total = (w["cls"] * losses["loss_cls"]
                 + w["bbox"] * losses["loss_bbox"]
                 + w["giou"] * losses["loss_giou"]
                 + w["m]()
