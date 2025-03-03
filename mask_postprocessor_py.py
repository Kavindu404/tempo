"""
D-FINE Instance Segmentation: Extended Post-Processor Implementation
Copyright (c) 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register
from .postprocessor import DFINEPostProcessor

@register()
class DFINEMaskPostProcessor(DFINEPostProcessor):
    __share__ = ["num_classes", "use_focal_loss", "num_top_queries", "remap_mscoco_category"]

    def __init__(
        self, 
        num_classes=80, 
        use_focal_loss=True, 
        num_top_queries=300, 
        mask_threshold=0.5,
        remap_mscoco_category=False
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            use_focal_loss=use_focal_loss,
            num_top_queries=num_top_queries,
            remap_mscoco_category=remap_mscoco_category
        )
        self.mask_threshold = mask_threshold

    def forward(self, outputs, orig_target_sizes):
        # First, get detection results from parent class
        results = super().forward(outputs, orig_target_sizes)
        
        # Process mask predictions if available
        if "pred_masks" in outputs:
            mask_logits = outputs["pred_masks"]  # [batch_size, num_queries, num_classes, mask_h, mask_w]
            
            # Get the detection results to match masks with correct classes and boxes
            for batch_idx, result in enumerate(results):
                # Get the class IDs and indices for each detection
                keep_indices = []
                for i, (score, label) in enumerate(zip(result["scores"], result["labels"])):
                    if score > self.score_threshold:
                        keep_indices.append(i)
                
                if not keep_indices:
                    # No detections survived threshold
                    result["masks"] = torch.zeros((0, orig_target_sizes[batch_idx][0], orig_target_sizes[batch_idx][1]), 
                                                device=mask_logits.device, dtype=torch.bool)
                    continue
                
                # Use topk indices to select corresponding masks
                # If using focal loss, we've already selected topk boxes/scores in the parent class
                if self.use_focal_loss:
                    # For focal loss, we need to map the selected indices back to the original boxes
                    selected_indices = keep_indices
                    selected_labels = result["labels"][selected_indices]
                    
                    # Get masks for selected detections
                    # [num_keep, num_classes, mask_h, mask_w] -> select mask for each class
                    batch_masks = mask_logits[batch_idx, selected_indices]
                    
                    # Select the mask for the predicted class [num_keep, mask_h, mask_w]
                    masks = torch.stack([batch_masks[i, label] for i, label in enumerate(selected_labels)])
                else:
                    # For softmax, the process is similar
                    selected_indices = keep_indices
                    selected_labels = result["labels"][selected_indices]
                    
                    batch_masks = mask_logits[batch_idx, selected_indices]
                    masks = torch.stack([batch_masks[i, label] for i, label in enumerate(selected_labels)])
                
                # Resize masks to original image size
                h, w = orig_target_sizes[batch_idx]
                masks = F.interpolate(masks.unsqueeze(1), size=(h, w), 
                                     mode="bilinear", align_corners=False).squeeze(1)
                
                # Apply threshold to get binary masks
                masks = masks > self.mask_threshold
                
                # Store masks in results
                result["masks"] = masks
        
        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
