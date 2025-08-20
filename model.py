import torch
import torch.nn as nn

class DINOv3Mask2Former(nn.Module):
    """
    Wraps the official DINOv3 segmentor (Mask2Former-style) via torch.hub.
    Expects forward() -> dict with:
      - pred_logits: [B,Q,C+1] (includes "no-object")
      - pred_masks:  [B,Q,H,W] (logits)
    """
    def __init__(self, repo_dir: str, hub_entry: str, num_classes: int,
                 head_weights: str | None, backbone_weights: str | None):
        super().__init__()
        self.segmentor = torch.hub.load(
            repo_or_dir=repo_dir,
            model=hub_entry,
            source='local',
            weights=head_weights,
            backbone_weights=backbone_weights,
            num_classes=num_classes
        )

    def freeze_backbone(self, frozen: bool = True):
        if hasattr(self.segmentor, "backbone"):
            for p in self.segmentor.backbone.parameters():
                p.requires_grad = not frozen

    def forward(self, images: torch.Tensor):
        out = self.segmentor(images)
        assert "pred_logits" in out and "pred_masks" in out, \
            "Segmentor must return {'pred_logits','pred_masks'}"
        return out
