"""
D-FINE-Mask Postprocessor: Process outputs from D-FINE-Mask for instance segmentation.
This extends the DFINEPostProcessor to handle instance masks.
Copyright (c) 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register
from .postprocessor import DFINEPostProcessor, mod


@register()
class DFINEMaskPostProcessor(DFINEPostProcessor):
    """
    Post-processor for D-FINE-Mask model outputs.
    Extends DFINEPostProcessor to include mask processing.
    """
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
        
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        """
        Process the outputs of the model to produce formatted results
        with bounding boxes and masks.
        
        Args:
            outputs: dict with model outputs including pred_logits, pred_boxes, and pred_masks
            orig_target_sizes: tensor of original image sizes [batch_size, 2]
            
        Returns:
            list of dicts with processed detection results
        """
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        
        # Get predicted masks if available
        has_masks = "pred_masks" in outputs
        if has_masks:
            masks = outputs["pred_masks"]
            
        # Process boxes the same way as in the parent class
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            )
            
            # Also gather the corresponding masks if available
            if has_masks:
                masks = masks.gather(
                    dim=1, index=index.unsqueeze(-1).unsqueeze(-1).repeat(
                        1, 1, masks.shape[2], masks.shape[3]
                    )
                )
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )
                
                # Also gather the corresponding masks if available
                if has_masks:
                    masks = masks.gather(
                        dim=1, index=index.unsqueeze(-1).unsqueeze(-1).repeat(
                            1, 1, masks.shape[2], masks.shape[3]
                        )
                    )
                    
        # For ONNX export mode
        if self.deploy_mode:
            if has_masks:
                return labels, boxes, scores, masks
            else:
                return labels, boxes, scores
                
        # Apply category remapping if requested
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category
            labels = (
                torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])
                .to(boxes.device)
                .reshape(labels.shape)
            )
            
        # Process masks: resize to original image size and apply threshold
        if has_masks:
            processed_masks = []
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                # Resize masks to original image size
                img_h, img_w = orig_target_sizes[i]
                mask = F.interpolate(
                    mask.unsqueeze(0),
                    size=(int(img_h.item()), int(img_w.item())),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)
                
                # Apply sigmoid and threshold
                mask = torch.sigmoid(mask) > self.mask_threshold
                processed_masks.append(mask)
        else:
            processed_masks = [None] * len(boxes)
                
        # Combine all outputs into result dictionaries
        results = []
        for lab, box, sco, msk in zip(labels, boxes, scores, processed_masks):
            result = dict(labels=lab, boxes=box, scores=sco)
            if msk is not None:
                result["masks"] = msk
            results.append(result)
            
        return results
        
    def extra_repr(self) -> str:
        """Add mask_threshold to string representation"""
        base_repr = super().extra_repr()
        return f"{base_repr}, mask_threshold={self.mask_threshold}"
