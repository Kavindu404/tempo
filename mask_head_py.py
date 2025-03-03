"""
D-FINE Instance Segmentation: Mask Head Implementation
Copyright (c) 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from ...core import register

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, act="relu"):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


@register()
class MaskHead(nn.Module):
    def __init__(self, hidden_dim=256, num_conv=4, num_classes=80, mask_resolution=28):
        super().__init__()
        
        self.mask_resolution = mask_resolution
        self.hidden_dim = hidden_dim
        
        # Convolutional layers for mask features
        self.mask_convs = nn.ModuleList()
        for _ in range(num_conv):
            self.mask_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, act="relu")
            )
        
        # Final mask prediction layer
        self.mask_pred = nn.Conv2d(hidden_dim, num_classes, 1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, boxes):
        """
        Args:
            x: List of FPN features [N, C, H, W]
            boxes: Predicted boxes [N, num_queries, 4] in xyxy format
        Returns:
            masks: Predicted masks [N, num_queries, num_classes, mask_h, mask_w]
        """
        # Choose appropriate feature level based on typical practice
        # Use level with stride=8 (typically index 0 or 1 in feature list)
        features = x[1]  # Using P3 features (stride=8)
        
        # Get batch size and number of queries
        batch_size = features.shape[0]
        num_queries = boxes.shape[1]
        
        # Extract ROI features for each box
        roi_features = self._extract_roi_features(features, boxes)
        
        # Process through mask convs
        for conv in self.mask_convs:
            roi_features = conv(roi_features)
        
        # Predict masks (N*num_queries, num_classes, mask_h, mask_w)
        mask_logits = self.mask_pred(roi_features)
        
        # Reshape to [N, num_queries, num_classes, mask_h, mask_w]
        mask_logits = mask_logits.reshape(
            batch_size, num_queries, -1, self.mask_resolution, self.mask_resolution
        )
        
        return mask_logits
    
    def _extract_roi_features(self, features, boxes):
        """
        Extract ROI features using ROI Align
        
        Args:
            features: FPN features [N, C, H, W]
            boxes: Boxes [N, num_queries, 4] in xyxy format
            
        Returns:
            roi_features: ROI features [N*num_queries, C, mask_h, mask_w]
        """
        batch_size = features.shape[0]
        num_queries = boxes.shape[1]
        
        # Convert boxes to format expected by roi_align
        # RoI align expects boxes in format [batch_idx, x1, y1, x2, y2]
        rois = []
        
        h, w = features.shape[2:]
        scale = 1.0 / 8.0  # P3 feature map scale factor
        
        for batch_idx in range(batch_size):
            # Scale boxes to feature map size
            # For COCO-formatted boxes (0-1 range), scale to feature map size
            batch_boxes = boxes[batch_idx].clone()
            batch_boxes[:, [0, 2]] *= w
            batch_boxes[:, [1, 3]] *= h
            
            # Add batch index to boxes
            batch_indices = torch.full((batch_boxes.shape[0], 1), batch_idx, 
                                      dtype=torch.float32, device=boxes.device)
            rois.append(torch.cat([batch_indices, batch_boxes], dim=1))
        
        rois = torch.cat(rois, dim=0)
        
        # Apply RoI align
        roi_features = roi_align(
            features, 
            rois, 
            output_size=(self.mask_resolution, self.mask_resolution),
            spatial_scale=scale, 
            sampling_ratio=2,
            aligned=True
        )
        
        return roi_features
