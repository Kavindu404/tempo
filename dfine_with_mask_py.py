"""
D-FINE Instance Segmentation: Extended Model Implementation
Copyright (c) 2024
"""

import torch
import torch.nn as nn

from ...core import register

@register()
class DFINEWithMask(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
        "mask_head",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        mask_head: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.mask_head = mask_head

    def forward(self, x, targets=None):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Process features through encoder
        encoded_features = self.encoder(features)
        
        # Get detection outputs from decoder
        detection_outputs = self.decoder(encoded_features, targets)
        
        # Add mask prediction
        if self.training:
            # During training, only compute masks if we have detection outputs
            pred_boxes = detection_outputs["pred_boxes"]
            
            # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format
            pred_boxes_xyxy = torch.cat([
                pred_boxes[..., 0:2] - pred_boxes[..., 2:4] / 2,
                pred_boxes[..., 0:2] + pred_boxes[..., 2:4] / 2
            ], dim=-1)
            
            # Predict masks
            mask_outputs = self.mask_head(features, pred_boxes_xyxy)
            detection_outputs["pred_masks"] = mask_outputs
            
            # Also add masks to auxiliary outputs
            if "aux_outputs" in detection_outputs:
                for i, aux_output in enumerate(detection_outputs["aux_outputs"]):
                    aux_boxes = aux_output["pred_boxes"]
                    aux_boxes_xyxy = torch.cat([
                        aux_boxes[..., 0:2] - aux_boxes[..., 2:4] / 2, 
                        aux_boxes[..., 0:2] + aux_boxes[..., 2:4] / 2
                    ], dim=-1)
                    
                    # We can compute masks for all aux outputs, but this is computational intensive
                    # For efficiency, we might want to compute masks only for the last aux output
                    if i == len(detection_outputs["aux_outputs"]) - 1:
                        aux_masks = self.mask_head(features, aux_boxes_xyxy)
                        aux_output["pred_masks"] = aux_masks
        else:
            # During inference
            pred_boxes = detection_outputs["pred_boxes"]
            
            # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format
            pred_boxes_xyxy = torch.cat([
                pred_boxes[..., 0:2] - pred_boxes[..., 2:4] / 2,
                pred_boxes[..., 0:2] + pred_boxes[..., 2:4] / 2
            ], dim=-1)
            
            # Predict masks
            mask_outputs = self.mask_head(features, pred_boxes_xyxy)
            detection_outputs["pred_masks"] = mask_outputs
            
        return detection_outputs

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
