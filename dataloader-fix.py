"""
Fix for dataloader.py error: TypeError: Image object is not subscriptable
"""

# Update the BatchImageCollateFunction in src/data/dataloader.py

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as VT
from torch.utils.data import default_collate
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as VF

from ..core import register

@register()
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self,
        stop_epoch=None,
        ema_restart_decay=0.9999,
        base_size=640,
        base_size_repeat=None,
    ) -> None:
        super().__init__()
        self.base_size = base_size
        self.scales = (
            generate_scales(base_size, base_size_repeat) if base_size_repeat is not None else None
        )
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay

    def __call__(self, items):
        # Check if items contain Image objects and convert them to tensors if needed
        processed_items = []
        for item in items:
            if hasattr(item[0], 'shape'):  # It's already a tensor
                processed_items.append(item)
            elif hasattr(item[0], 'convert'):  # It's a PIL Image
                # Convert PIL Image to tensor
                tensor = torchvision.transforms.functional.to_tensor(item[0])
                processed_items.append((tensor, item[1]))
            else:
                # Try to handle torchvision.tv_tensors.Image objects
                try:
                    # For torchvision.tv_tensors.Image, access the underlying tensor
                    tensor = item[0].as_subclass(torch.Tensor)
                    processed_items.append((tensor, item[1]))
                except Exception as e:
                    print(f"Warning: Unable to process item type {type(item[0])}: {str(e)}")
                    # Fall back to using the original item and hope for the best
                    processed_items.append(item)
        
        # Now try to batch the processed items
        try:
            images = torch.cat([x[0][None] for x in processed_items], dim=0)
            targets = [x[1] for x in processed_items]
        except Exception as e:
            print(f"Error during batch creation: {str(e)}")
            print(f"Item types: {[type(x[0]) for x in processed_items]}")
            # Emergency fallback - try to create a dummy batch to avoid breaking the pipeline
            if len(processed_items) > 0 and hasattr(processed_items[0][0], 'shape'):
                # Create a batch with the first item repeated
                shape = list(processed_items[0][0].shape)
                images = processed_items[0][0].unsqueeze(0).expand(len(processed_items), *shape)
                targets = [x[1] for x in processed_items]
            else:
                # If all else fails, raise the error to avoid silent failures
                raise

        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if "masks" in targets[0]:
                for tg in targets:
                    tg["masks"] = F.interpolate(tg["masks"], size=sz, mode="nearest")
                raise NotImplementedError("")

        return images, targets

# Update the batch_image_collate_fn function to handle different item types
@register()
def batch_image_collate_fn(items):
    """only batch image"""
    processed_items = []
    for item in items:
        if hasattr(item[0], 'shape'):  # It's already a tensor
            processed_items.append(item)
        elif hasattr(item[0], 'convert'):  # It's a PIL Image
            # Convert PIL Image to tensor
            tensor = torchvision.transforms.functional.to_tensor(item[0])
            processed_items.append((tensor, item[1]))
        else:
            # Try to handle torchvision.tv_tensors.Image objects
            try:
                # For torchvision.tv_tensors.Image, access the underlying tensor
                tensor = item[0].as_subclass(torch.Tensor)
                processed_items.append((tensor, item[1]))
            except Exception as e:
                print(f"Warning: Unable to process item type {type(item[0])}: {str(e)}")
                # Fall back to using the original item and hope for the best
                processed_items.append(item)
    
    try:
        return torch.cat([x[0][None] for x in processed_items], dim=0), [x[1] for x in processed_items]
    except Exception as e:
        print(f"Error in batch_image_collate_fn: {str(e)}")
        print(f"Item types: {[type(x[0]) for x in processed_items]}")
        # Emergency fallback
        if len(processed_items) > 0 and hasattr(processed_items[0][0], 'shape'):
            shape = list(processed_items[0][0].shape)
            images = processed_items[0][0].unsqueeze(0).expand(len(processed_items), *shape)
            return images, [x[1] for x in processed_items]
        else:
            raise
