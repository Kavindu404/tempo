"""
D-FINE-Mask: Extension of D-FINE model for instance segmentation.
This module adds mask prediction capabilities to the D-FINE object detection model.
Copyright (c) 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torch import Tensor

from ...core import register


class MaskHead(nn.Module):
    """
    Simple convolutional head for generating instance masks.
    Takes features from the decoder and produces binary masks.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        mask_dim: int = 256,
        num_convs: int = 4,
        mask_resolution: int = 28,
    ):
        super().__init__()
        
        # Create a small FCN for mask prediction
        self.mask_resolution = mask_resolution
        self.mask_dim = mask_dim
        self.hidden_dim = hidden_dim
        
        # Build the mask head with multiple convolutional layers
        layers = []
        for i in range(num_convs):
            in_channels = hidden_dim if i == 0 else mask_dim
            layers.append(nn.Conv2d(in_channels, mask_dim, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        
        # Final layer to predict binary mask
        layers.append(nn.Conv2d(mask_dim, mask_dim, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(mask_dim, 1, 1))
        
        self.mask_head = nn.Sequential(*layers)
        
        # Weight initialization
        for m in self.mask_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, num_queries, hidden_dim]
        
        Returns:
            masks: Tensor of shape [batch_size, num_queries, mask_resolution, mask_resolution]
        """
        batch_size, num_queries, hidden_dim = x.shape
        
        # Reshape to pass through the convolutional layers
        x = x.reshape(batch_size * num_queries, hidden_dim, 1, 1)
        
        # Upsample to the mask resolution
        x = F.interpolate(x, size=(self.mask_resolution, self.mask_resolution), mode='bilinear', align_corners=False)
        
        # Process through mask head
        masks = self.mask_head(x)
        
        # Reshape output
        masks = masks.view(batch_size, num_queries, self.mask_resolution, self.mask_resolution)
        
        return masks


@register()
class DFINEMask(nn.Module):
    """
    D-FINE-Mask: Extension of D-FINE for instance segmentation.
    Adds a mask prediction head to the DFINE architecture.
    """
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
        mask_head: nn.Module = None,
        mask_dim: int = 256,
        mask_resolution: int = 28,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        
        if mask_head is None:
            # Create default mask head if not provided
            hidden_dim = getattr(decoder, "hidden_dim", 256)
            self.mask_head = MaskHead(
                hidden_dim=hidden_dim,
                mask_dim=mask_dim,
                mask_resolution=mask_resolution
            )
        else:
            self.mask_head = mask_head
            
    def forward(self, x, targets=None):
        """
        Forward pass of the DFINEMask model.
        
        Args:
            x: Input image tensor
            targets: Optional dictionary of ground-truth targets
        
        Returns:
            Dictionary containing detection and instance segmentation outputs
        """
        x = self.backbone(x)
        x = self.encoder(x)
        outputs = self.decoder(x, targets)
        
        # Extract features to generate masks
        if self.training:
            # During training, use features from all decoder layers
            mask_features = outputs['aux_outputs'][-1]['pred_logits']
        else:
            # During inference, just use the final layer
            mask_features = outputs['pred_logits']
            
        # Generate masks from features
        outputs['pred_masks'] = self.mask_head(mask_features)
        
        # Add masks to auxiliary outputs if in training mode
        if self.training and 'aux_outputs' in outputs:
            for i, aux_output in enumerate(outputs['aux_outputs']):
                aux_features = aux_output['pred_logits']
                outputs['aux_outputs'][i]['pred_masks'] = self.mask_head(aux_features)
                
        return outputs

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
