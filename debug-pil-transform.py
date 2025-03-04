"""
Enhanced debugging for ConvertPILImage transform
"""

import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import traceback
from typing import Any, Dict

from ...core import register
from .._misc import (
    BoundingBoxes,
    Image,
    Mask,
    SanitizeBoundingBoxes as TVSanitizeBoundingBoxes,
    Video,
    _boxes_keys,
    convert_to_tv_tensor,
)

@register()
class ConvertPILImage(nn.Module):
    """
    Fall back to a simpler implementation using nn.Module instead of T.Transform
    to bypass the torchvision v2 API issues.
    """
    def __init__(self, dtype="float32", scale=True):
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def forward(self, x):
        try:
            if isinstance(x, PIL.Image.Image):
                # Convert PIL Image to tensor
                img = torchvision.transforms.functional.to_tensor(x)  # Using v1 API
                
                if self.dtype == "float32":
                    img = img.float()
                
                # Scale is already handled by to_tensor for PIL images
                
                return img  # Return plain tensor instead of Image wrapper
            
            # If it's already a tensor or another type, return as is
            return x
        except Exception as e:
            print(f"Error in ConvertPILImage: {str(e)}")
            print(f"Input type: {type(x)}")
            print(traceback.format_exc())
            raise

# Alternative implementation using try-except with the v2 API
@register()
class ConvertPILImageV2(T.Transform):
    def __init__(self, dtype="float32", scale=True):
        super().__init__()
        self.dtype = dtype
        self.scale = scale
        
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        try:
            if isinstance(inpt, PIL.Image.Image):
                # Convert PIL Image to tensor
                tensor = F.pil_to_tensor(inpt)
                
                if self.dtype == "float32":
                    tensor = tensor.float()
                
                if self.scale:
                    tensor = tensor / 255.0
                
                # For debugging
                print(f"ConvertPILImageV2: Successfully converted {type(inpt)} to tensor of shape {tensor.shape}")
                
                return tensor  # Return plain tensor instead of Image wrapper
            
            # Return other types unchanged
            return inpt
        except Exception as e:
            print(f"Error in ConvertPILImageV2._transform: {str(e)}")
            print(f"Input type: {type(inpt)}")
            print(traceback.format_exc())
            raise
            
    def forward(self, *inputs):
        try:
            # For non-list/tuple single inputs
            if len(inputs) == 1 and not isinstance(inputs[0], (list, tuple)):
                return self._transform(inputs[0], {})
            
            # Handle different input formats
            if len(inputs) == 3 and isinstance(inputs[0], PIL.Image.Image):
                # This is likely (image, target, dataset) format
                image, target, dataset = inputs
                transformed_image = self._transform(image, {})
                return transformed_image, target, dataset
            
            # Default handling
            return super().forward(*inputs)
        except Exception as e:
            print(f"Error in ConvertPILImageV2.forward: {str(e)}")
            print(f"Inputs: {[type(inp) for inp in inputs]}")
            print(traceback.format_exc())
            raise

# Fix for Container.py to add better error handling
def stop_epoch_forward(self, *inputs):
    """
    Enhanced error handling for stop_epoch_forward method in Compose class
    """
    sample = inputs if len(inputs) > 1 else inputs[0]
    dataset = sample[-1] if isinstance(sample, tuple) and len(sample) > 2 else None
    cur_epoch = getattr(dataset, 'epoch', -1) if dataset is not None else -1
    policy_ops = self.policy.get("ops", [])
    policy_epoch = self.policy.get("epoch", float('inf'))

    print(f"Processing with stop_epoch_forward, current epoch: {cur_epoch}, policy_epoch: {policy_epoch}")
    print(f"Policy ops: {policy_ops}")
    
    for i, transform in enumerate(self.transforms):
        transform_name = type(transform).__name__
        print(f"Applying transform {i}: {transform_name}")
        
        # Skip transforms mentioned in policy_ops if we're past the policy_epoch
        if transform_name in policy_ops and cur_epoch >= policy_epoch:
            print(f"Skipping transform {transform_name} due to epoch policy")
            continue
            
        try:
            sample = transform(sample)
            print(f"Successfully applied transform {transform_name}")
        except Exception as e:
            print(f"Error in transform {transform_name}: {str(e)}")
            print(f"Sample type: {type(sample)}")
            if isinstance(sample, tuple):
                print(f"Sample element types: {[type(s) for s in sample]}")
            print(traceback.format_exc())
            raise
    
    return sample
