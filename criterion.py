from typing import List, Tuple
import torch
from torch import nn
import torch.nn.functional as F

def sigmoid_dice_loss(inputs, targets, eps=1e-6):
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.float().flatten(1)
    inter = (inputs * targets).sum(1)
    union = inputs.sum(1) + targets.sum(1)
    return 1 - (2*inter + eps) / (union + eps)

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict=None, eos_coef=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict or {"loss_ce":1.0, "loss_mask":5.0, "loss_dice":5.0}
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i,(src,_) in enumerate(indices)])
        src_idx = torch.cat([src for (src,_) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs["pred_logits"]                       # [B,Q,C+1]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t,(_,J) in zip(targets, indices)], 0)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.long, device=src_logits.device)
        target_classes[idx] = target_classes_o
        return {"loss_ce": F.cross_entropy(src_logits.transpose(1,2), target_classes, self.empty_weight)}

    def loss_masks(self, outputs, targets, indices):
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]               # [N,H,W] logits
        tgt_masks = torch.cat([t["masks"][i] for t,(_,i) in zip(targets, indices)], 0)  # [N,H,W]
        loss_mask = F.l1_loss(src_masks.sigmoid(), tgt_masks.float(), reduction="mean")
        loss_dice = sigmoid_dice_loss(src_masks, tgt_masks).mean()
        return {"loss_mask": loss_mask, "loss_dice": loss_dice}

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_masks(outputs, targets, indices))
        total = sum(self.weight_dict[k]*v for k,v in losses.items())
        losses["loss_total"] = total
        return losses, indices
