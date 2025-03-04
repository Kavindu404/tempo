"""
Fix for ConvertBoxes transform to handle tuple input properly
"""

import torch
import torch.nn as nn
import torchvision.ops
from ...core import register
from .._misc import convert_to_tv_tensor, BoundingBoxes, _boxes_keys

@register()
class ConvertBoxes(nn.Module):
    """
    Simple transform to convert bounding boxes formats.
    Uses traditional nn.Module instead of torchvision.transforms.v2 to avoid API issues.
    """
    def __init__(self, fmt="", normalize=False):
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def forward(self, *args):
        # Handle different input formats
        if len(args) == 1:
            # Single input case
            x = args[0]
            return self._process_input(x)
        elif len(args) == 3:
            # (image, target, dataset) format
            image, target, dataset = args
            
            # Only process the target dict containing the boxes
            if isinstance(target, dict) and "boxes" in target:
                target["boxes"] = self._process_boxes(target["boxes"])
            
            return image, target, dataset
        
        # Return inputs unchanged for any other format
        return args
    
    def _process_input(self, x):
        """Process a single input which could be a BoundingBoxes object or a dict"""
        if isinstance(x, BoundingBoxes):
            return self._process_boxes(x)
        elif isinstance(x, dict) and "boxes" in x:
            x["boxes"] = self._process_boxes(x["boxes"])
        return x
    
    def _process_boxes(self, boxes):
        """Process BoundingBoxes tensor"""
        if not isinstance(boxes, BoundingBoxes) and not torch.is_tensor(boxes):
            # If not a BoundingBoxes object or tensor, return unchanged
            return boxes
            
        if self.fmt:
            # If it's a BoundingBoxes object, extract format and spatial_size
            if isinstance(boxes, BoundingBoxes):
                in_fmt = boxes.format.value.lower()
                spatial_size = getattr(boxes, _boxes_keys[1])
            else:
                # Default values if not a BoundingBoxes object
                in_fmt = "xyxy"
                spatial_size = None
                
            # Convert box format
            boxes_tensor = torchvision.ops.box_convert(boxes, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            
            # Convert to TV tensor if spatial_size is available
            if spatial_size is not None:
                boxes_tensor = convert_to_tv_tensor(
                    boxes_tensor, key="boxes", box_format=self.fmt.upper(), spatial_size=spatial_size
                )
            
            # Apply normalization if requested
            if self.normalize and spatial_size is not None:
                if isinstance(spatial_size, torch.Tensor):
                    boxes_tensor = boxes_tensor / spatial_size.repeat(2)[None]
                else:
                    boxes_tensor = boxes_tensor / torch.tensor(spatial_size).repeat(2)[None]
            
            return boxes_tensor
            
        return boxes
