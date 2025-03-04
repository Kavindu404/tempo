"""
Fix for torchvision.transforms.v2 compatibility in D-FINE.

This file provides proper implementations of _transform methods for custom transforms
to work with torchvision.transforms.v2 framework.
"""

# Update src/data/transforms/_transforms.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

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

torchvision.disable_beta_transforms_warning()

# These transforms already work with v2 API
RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register(name="SanitizeBoundingBoxes")
class SanitizeBoundingBoxes(TVSanitizeBoundingBoxes):
    """Use the built-in SanitizeBoundingBoxes from torchvision which already implements _transform correctly"""
    pass


@register()
class EmptyTransform(T.Transform):
    def __init__(self):
        super().__init__()

    def _transform(self, inpt, params):
        # Proper implementation of _transform for v2 API
        return inpt

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Transform):
    _transformed_types = (
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def __init__(self, size, fill=0, padding_mode="constant"):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def _get_params(self, flat_inputs):
        for i in flat_inputs:
            if hasattr(i, "shape"):
                sp = i.shape[-2:]
                break
            if hasattr(i, "size"):
                sp = i.size[::-1]  # PIL images use (width, height)
                break
        
        h, w = self.size[0] - sp[0], self.size[1] - sp[1]
        padding = [0, 0, w, h]
        return {"padding": padding}

    def _transform(self, inpt, params):
        # Implementation for v2 API
        padding = params["padding"]
        return F.pad(inpt, padding=padding, fill=self.fill, padding_mode=self.padding_mode)


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials
        )
        self.p = p

    def forward(self, *inputs):
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        
        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt="", normalize=False):
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt, params):
        if isinstance(inpt, BoundingBoxes):
            spatial_size = getattr(inpt, _boxes_keys[1])
            if self.fmt:
                in_fmt = inpt.format.value.lower()
                transformed = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
                transformed = convert_to_tv_tensor(
                    transformed, key="boxes", box_format=self.fmt.upper(), spatial_size=spatial_size
                )
            else:
                transformed = inpt
                
            if self.normalize:
                if isinstance(spatial_size, torch.Tensor):
                    transformed = transformed / spatial_size.repeat(2)[None]
                else:
                    transformed = transformed / torch.tensor(spatial_size).repeat(2)[None]
                    
            return transformed
        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (Image.Image, )  # PIL.Image.Image

    def __init__(self, dtype="float32", scale=True):
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt, params):
        # Handle PIL images
        img = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            img = img.float()

        if self.scale:
            img = img / 255.0

        return Image(img)


# Update src/data/transforms/container.py
# Fix the Compose class

class Compose(T.Compose):
    def __init__(self, ops, policy=None):
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop("type")
                    transform = getattr(
                        GLOBAL_CONFIG[name]["_pymodule"], GLOBAL_CONFIG[name]["_name"]
                    )(**op)
                    transforms.append(transform)
                    op["type"] = name
                elif isinstance(op, nn.Module):
                    transforms.append(op)
                else:
                    raise ValueError(f"Unsupported transform type: {type(op)}")
        else:
            transforms = [EmptyTransform()]

        super().__init__(transforms=transforms)

        if policy is None:
            policy = {"name": "default"}

        self.policy = policy
        self.global_samples = 0

    def forward(self, *inputs):
        return self.get_forward(self.policy["name"])(*inputs)

    def get_forward(self, name):
        forwards = {
            "default": self.default_forward,
            "stop_epoch": self.stop_epoch_forward,
            "stop_sample": self.stop_sample_forward,
        }
        return forwards[name]

    def default_forward(self, *inputs):
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def stop_epoch_forward(self, *inputs):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1] if len(sample) > 2 else None
        cur_epoch = getattr(dataset, 'epoch', -1) if dataset is not None else -1
        policy_ops = self.policy.get("ops", [])
        policy_epoch = self.policy.get("epoch", float('inf'))

        for transform in self.transforms:
            # Skip transforms mentioned in policy_ops if we're past the policy_epoch
            if type(transform).__name__ in policy_ops and cur_epoch >= policy_epoch:
                continue
            try:
                sample = transform(sample)
            except Exception as e:
                print(f"Error in transform {type(transform).__name__}: {str(e)}")
                raise
        
        return sample

    def stop_sample_forward(self, *inputs):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]
        policy_ops = self.policy.get("ops", [])
        policy_sample = self.policy.get("sample", float('inf'))

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and self.global_samples >= policy_sample:
                continue
            
            try:
                sample = transform(sample)
            except Exception as e:
                print(f"Error in transform {type(transform).__name__}: {str(e)}")
                raise

        self.global_samples += 1
        return sample
