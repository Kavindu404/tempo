# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Mask2Former repository
# by Facebook, Inc. and its affiliates, used under the Apache 2.0 License.
# ---------------------------------------------------------------

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import io
import wandb

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule


class MaskClassificationInstance(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        lr: float = 1e-4,
        llrd: float = 0.8,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        eval_top_k_instances: int = 100,
        plot_every_n_batches: int = 50,  # New parameter for plotting frequency
        ckpt_path: Optional[str] = None,
        load_ckpt_class_head: bool = True,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes: List[int] = []
        self.eval_top_k_instances = eval_top_k_instances
        self.plot_every_n_batches = plot_every_n_batches

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

        self.init_metrics_instance(self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1)

    @torch.compiler.disable
    def plot_instance_results(
        self,
        img: torch.Tensor,
        target: dict,
        pred: dict,
        log_prefix: str,
        block_idx: int,
        batch_idx: int,
        confidence_threshold: float = 0.5,
        max_instances: int = 20,
    ):
        """Plot instance segmentation results with ground truth and predictions"""
        
        # Convert image to numpy
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot original image
        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Plot ground truth
        axes[1].imshow(img_np)
        gt_masks = target["masks"].cpu().numpy()
        gt_labels = target["labels"].cpu().numpy()
        
        # Generate colors for different instances
        colors = plt.cm.Set3(np.linspace(0, 1, len(gt_masks)))
        
        for i, (mask, label) in enumerate(zip(gt_masks, gt_labels)):
            if mask.sum() > 0:  # Only plot non-empty masks
                # Create colored mask
                colored_mask = np.zeros((*mask.shape, 4))
                colored_mask[mask > 0] = colors[i % len(colors)]
                colored_mask[:, :, 3] = 0.6 * mask  # Set alpha
                
                axes[1].imshow(colored_mask)
                
                # Add bounding box
                coords = np.where(mask)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    rect = patches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=2, edgecolor=colors[i % len(colors)], facecolor='none'
                    )
                    axes[1].add_patch(rect)
                    
                    # Add label
                    axes[1].text(
                        x_min, y_min - 5, f'GT: {label}',
                        color=colors[i % len(colors)], fontsize=8, fontweight='bold'
                    )
        
        axes[1].set_title(f"Ground Truth ({len(gt_masks)} instances)")
        axes[1].axis('off')
        
        # Plot predictions
        axes[2].imshow(img_np)
        pred_masks = pred["masks"].cpu().numpy()
        pred_labels = pred["labels"].cpu().numpy()
        pred_scores = pred["scores"].cpu().numpy()
        
        # Filter predictions by confidence
        keep_indices = pred_scores > confidence_threshold
        pred_masks = pred_masks[keep_indices]
        pred_labels = pred_labels[keep_indices]
        pred_scores = pred_scores[keep_indices]
        
        # Sort by confidence and keep top instances
        if len(pred_scores) > 0:
            sorted_indices = np.argsort(pred_scores)[::-1][:max_instances]
            pred_masks = pred_masks[sorted_indices]
            pred_labels = pred_labels[sorted_indices]
            pred_scores = pred_scores[sorted_indices]
        
        # Generate colors for predictions
        pred_colors = plt.cm.Set1(np.linspace(0, 1, len(pred_masks)))
        
        for i, (mask, label, score) in enumerate(zip(pred_masks, pred_labels, pred_scores)):
            if mask.sum() > 0:  # Only plot non-empty masks
                # Create colored mask
                colored_mask = np.zeros((*mask.shape, 4))
                colored_mask[mask > 0] = pred_colors[i % len(pred_colors)]
                colored_mask[:, :, 3] = 0.6 * mask  # Set alpha
                
                axes[2].imshow(colored_mask)
                
                # Add bounding box
                coords = np.where(mask)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    rect = patches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=2, edgecolor=pred_colors[i % len(pred_colors)], facecolor='none'
                    )
                    axes[2].add_patch(rect)
                    
                    # Add label with confidence
                    axes[2].text(
                        x_min, y_min - 5, f'{label}: {score:.2f}',
                        color=pred_colors[i % len(pred_colors)], fontsize=8, fontweight='bold'
                    )
        
        axes[2].set_title(f"Predictions ({len(pred_masks)} instances, conf > {confidence_threshold})")
        axes[2].axis('off')
        
        # Save and log to wandb
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        block_postfix = self.block_postfix(block_idx)
        name = f"{log_prefix}_instance_results_batch_{batch_idx}{block_postfix}"
        
        if hasattr(self.trainer.logger, 'experiment'):
            self.trainer.logger.experiment.log({name: [wandb.Image(Image.open(buf))]})

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(transformed_imgs)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
                mask_logits, img_sizes
            )

            preds, targets_ = [], []
            for j in range(len(mask_logits)):
                scores = class_logits[j].softmax(dim=-1)[:, :-1]
                labels = (
                    torch.arange(scores.shape[-1], device=self.device)
                    .unsqueeze(0)
                    .repeat(scores.shape[0], 1)
                    .flatten(0, 1)
                )

                topk_scores, topk_indices = scores.flatten(0, 1).topk(
                    self.eval_top_k_instances, sorted=False
                )
                labels = labels[topk_indices]

                topk_indices = topk_indices // scores.shape[-1]
                mask_logits[j] = mask_logits[j][topk_indices]

                masks = mask_logits[j] > 0
                mask_scores = (
                    mask_logits[j].sigmoid().flatten(1) * masks.flatten(1)
                ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
                scores = topk_scores * mask_scores

                preds.append(
                    dict(
                        masks=masks,
                        labels=labels,
                        scores=scores,
                    )
                )
                targets_.append(
                    dict(
                        masks=targets[j]["masks"],
                        labels=targets[j]["labels"],
                        iscrowd=targets[j]["is_crowd"],
                    )
                )

            self.update_metrics_instance(preds, targets_, i)
            
            # Plot results for the first image in batch at specified intervals
            if (batch_idx is not None and 
                batch_idx % self.plot_every_n_batches == 0 and 
                log_prefix == "val" and 
                len(preds) > 0):
                
                self.plot_instance_results(
                    imgs[0], targets[0], preds[0], log_prefix, i, batch_idx
                )

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_instance("val")

    def on_validation_end(self):
        self._on_eval_end_instance("val")