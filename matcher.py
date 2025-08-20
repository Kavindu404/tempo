from typing import List, Tuple
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

def _dice_cost(pred_logits_masks, tgt_masks, eps=1e-6):
    probs = pred_logits_masks.sigmoid().flatten(1)   # [Q,HW]
    tgt = tgt_masks.flatten(1).float()               # [T,HW]
    inter = probs @ tgt.T
    sums = probs.sum(-1).unsqueeze(1) + tgt.sum(-1).unsqueeze(0)
    return 1 - (2 * inter + eps) / (sums + eps)      # [Q,T]

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_mask=5.0, cost_dice=5.0):
        super().__init__()
        self.cc, self.cm, self.cd = cost_class, cost_mask, cost_dice

    @torch.no_grad()
    def forward(self, outputs: dict, targets: List[dict]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        prob = outputs["pred_logits"].softmax(-1)    # [B,Q,C+1]
        masks = outputs["pred_masks"]                # [B,Q,H,W]
        out = []
        for b in range(prob.size(0)):
            tgt_ids = targets[b]["labels"]
            tgt_masks = targets[b]["masks"].to(masks.dtype)
            if tgt_ids.numel() == 0:
                out.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            cost_class = -prob[b][:, tgt_ids]                # [Q,T]
            pm = masks[b].sigmoid().flatten(1)
            tm = tgt_masks.flatten(1)
            cost_mask = torch.cdist(pm, tm.float(), p=1)
            cost_dice = _dice_cost(masks[b], tgt_masks)
            C = self.cc*cost_class + self.cm*cost_mask + self.cd*cost_dice
            i, j = linear_sum_assignment(C.cpu())
            out.append((torch.as_tensor(i, dtype=torch.long), torch.as_tensor(j, dtype=torch.long)))
        return out
